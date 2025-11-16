import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score
import os
from typing import List, Tuple, Optional
import json
from .dnn import DNNHead, BertDNN
from .text_dataset import TextDataset
from torch.utils.data import random_split
import wandb

class BertDNNPipeline:
    def __init__(
        self,
        bert_model_name: str = 'bert-base-uncased',
        head_hidden_dims: List[int] = [512, 512, 256, 128, 64, 32, 16],
        num_classes: int = 4,
        max_length: int = 256,
        freeze_bert: bool = False,
        batch_size: int = 16,
        lr: float = 2e-4,
        device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        use_wandb: bool = True,
        wandb_project: str = "bert-dnn-pipeline",
        wandb_run_name: Optional[str] = None
    ):
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(
                project=wandb_project, 
                name=wandb_run_name,
                config={
                    "bert_model": bert_model_name,
                    "head_hidden_dims": head_hidden_dims,
                    "num_classes": num_classes,
                    "max_length": max_length,
                    "freeze_bert": freeze_bert,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "device": device
                }
            )
        print(f"Using device: {device}")
        self.device = torch.device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.lr = lr

        # Tokenizer & BERT
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        bert = BertModel.from_pretrained(bert_model_name)

        # Head
        bert_dim = bert.config.hidden_size  # 768
        head = DNNHead(bert_dim, head_hidden_dims, num_classes)

        # Full model
        self.model = BertDNN(bert, head, freeze_bert=freeze_bert).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = None  # Will be initialized in fit() if needed

    def fit(
        self,
        epochs: int = 3,
        verbose: bool = True,
        save_checkpoints: bool = True,
        checkpoint_dir: str = "./checkpoints",
        save_best_only: bool = True,
        use_scheduler: bool = True,
        scheduler_type: str = "cosine",  # Options: "cosine", "step", "plateau", "linear_warmup"
        scheduler_params: Optional[dict] = None
    ):
        train_dataset = TextDataset(mode="train", tokenizer=self.tokenizer, max_length=self.max_length)
        
        # Split the dataset into train and validation
        train_ratio = 0.8  # 80% for training
        train_size = int(len(train_dataset) * train_ratio)
        val_size = len(train_dataset) - train_size # Ensure sum equals total length

        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize learning rate scheduler
        if use_scheduler:
            self.scheduler = self._create_scheduler(scheduler_type, epochs, scheduler_params)
            if verbose:
                print(f"Using {scheduler_type} learning rate scheduler")
        
        best_val_acc = 0.0
        self.model.train()
        global_step = 0  # Track total training steps across all epochs
        
        for epoch in range(1, epochs + 1):
            total_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Create tokenized_phrase dict for model
                tokenized_phrase = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }
                logits = self.model(tokenized_phrase)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                global_step += 1
                
                # Log training loss at every step
                if self.use_wandb:
                    wandb.log({
                        "train_loss_step": loss.item(),
                        "step": global_step
                    })

            avg_loss = total_loss / len(train_loader)
            val_acc = self.evaluate(val_loader) if val_loader else None

            # Step the learning rate scheduler
            if self.scheduler is not None:
                if scheduler_type == "plateau":
                    # ReduceLROnPlateau needs a metric
                    self.scheduler.step(avg_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log validation accuracy and average epoch loss
            if self.use_wandb:
                log_dict = {
                    "epoch": epoch,
                    "train_loss_epoch": avg_loss,  # Average loss for the epoch
                    "learning_rate": current_lr,  # Log current learning rate
                }
                if val_acc is not None:
                    log_dict["val_accuracy"] = val_acc
                wandb.log(log_dict)

            if verbose:
                acc_str = f" | val_acc: {val_acc:.4f}" if val_acc else ""
                lr_str = f" | lr: {current_lr:.2e}"
                print(f"Epoch {epoch}/{epochs} | loss: {avg_loss:.4f}{acc_str}{lr_str}")
            
            # Save checkpoints
            if save_checkpoints:
                if save_best_only:
                    # Save only when validation accuracy improves
                    if val_acc and val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_path = os.path.join(checkpoint_dir, "best_model")
                        self.save_checkpoint(best_path, epoch, avg_loss, val_acc)
                        if verbose:
                            print(f"✓ New best model saved! val_acc: {val_acc:.4f}")
                else:
                    # Save every epoch
                    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
                    self.save_checkpoint(checkpoint_path, epoch, avg_loss, val_acc)
                    if verbose:
                        print(f"✓ Checkpoint saved at epoch {epoch}")
        
        if verbose and save_checkpoints:
            print(f"\nTraining completed! Best val_acc: {best_val_acc:.4f}")
        
        # Log final best validation accuracy to wandb
        if self.use_wandb:
            wandb.log({"best_val_accuracy": best_val_acc})

    def _create_scheduler(self, scheduler_type: str, epochs: int, params: Optional[dict] = None):
        """Create a learning rate scheduler based on type"""
        params = params or {}
        
        if scheduler_type == "cosine":
            # Cosine annealing: smoothly decrease LR to min_lr
            T_max = params.get("T_max", epochs)
            eta_min = params.get("eta_min", 1e-7)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=T_max, 
                eta_min=eta_min
            )
        
        elif scheduler_type == "step":
            # Step decay: reduce LR by gamma every step_size epochs
            step_size = params.get("step_size", max(1, epochs // 3))
            gamma = params.get("gamma", 0.1)
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=step_size, 
                gamma=gamma
            )
        
        elif scheduler_type == "plateau":
            # Reduce on plateau: reduce when metric stops improving
            mode = params.get("mode", "min")
            factor = params.get("factor", 0.5)
            patience = params.get("patience", 3)
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode=mode, 
                factor=factor, 
                patience=patience,
                verbose=True
            )
        
        elif scheduler_type == "linear_warmup":
            # Linear warmup followed by linear decay
            warmup_epochs = params.get("warmup_epochs", max(1, epochs // 10))
            start_factor = params.get("start_factor", 0.1)
            end_factor = params.get("end_factor", 0.01)
            
            # Warmup scheduler
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            
            # Decay scheduler
            decay = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=end_factor,
                total_iters=epochs - warmup_epochs
            )
            
            # Sequential scheduler
            return torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, decay],
                milestones=[warmup_epochs]
            )
        
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        preds, trues = [], []
        for batch in val_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels']

            tokenized_phrase = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            logits = self.model(tokenized_phrase)
            pred = logits.argmax(dim=1).cpu().numpy()
            true = labels.numpy()
            preds.extend(pred)
            trues.extend(true)
        return accuracy_score(trues, preds)

    @torch.no_grad()
    def predict(self, texts: List[str]) -> List[int]:
        self.model.eval()
        # Create a simple dataset for prediction
        preds = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            tokenized = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = tokenized['input_ids'].to(self.device)
            attention_mask = tokenized['attention_mask'].to(self.device)
            
            tokenized_phrase = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            logits = self.model(tokenized_phrase)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.extend(pred.tolist())
        return preds

    def save(self, path: str):
        """Save model for inference (model weights + tokenizer + config only)"""
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
        self.tokenizer.save_pretrained(path)

        config = {
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "lr": self.lr
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)
    
    def save_checkpoint(self, path: str, epoch: int, loss: float, val_acc: Optional[float] = None):
        """Save full training checkpoint including optimizer state"""
        os.makedirs(path, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'val_acc': val_acc,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'lr': self.lr
        }
        
        torch.save(checkpoint, os.path.join(path, "checkpoint.pt"))
        self.tokenizer.save_pretrained(path)
        
        # Also save a metadata file for easy inspection
        metadata = {
            'epoch': epoch,
            'loss': loss,
            'val_acc': val_acc
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str) -> Tuple[int, float, Optional[float]]:
        """
        Load checkpoint to resume training
        
        Returns:
            tuple: (epoch, loss, val_acc) from the checkpoint
        """
        checkpoint_file = os.path.join(checkpoint_path, "checkpoint.pt")
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        val_acc = checkpoint.get('val_acc')
        
        val_acc_str = f"{val_acc:.4f}" if val_acc is not None else "N/A"
        print(f"Checkpoint loaded: epoch {epoch}, loss: {loss:.4f}, val_acc: {val_acc_str}")
        return epoch, loss, val_acc

    @classmethod
    def load(cls, path: str, head_hidden_dims=[512, 256], num_classes=4, freeze_bert=False):
        tokenizer = BertTokenizer.from_pretrained(path)
        bert = BertModel.from_pretrained('bert-base-uncased')  # reload base

        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)

        head = DNNHead(bert.config.hidden_size, head_hidden_dims, num_classes)
        model = BertDNN(bert, head, freeze_bert=freeze_bert)
        model.load_state_dict(torch.load(os.path.join(path, "model.pt"), map_location='cpu'))

        pipeline = cls(
            bert_model_name='bert-base-uncased',
            head_hidden_dims=head_hidden_dims,
            num_classes=num_classes,
            max_length=config["max_length"],
            freeze_bert=freeze_bert,
            batch_size=config["batch_size"],
            lr=config["lr"]
        )
        pipeline.model = model.to(pipeline.device)
        pipeline.tokenizer = tokenizer
        return pipeline