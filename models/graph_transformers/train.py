def train_epoch(model, graph, loader, optimizer, criterion, device, neg_samples=1):
    """
    Train model for one epoch.
    
    Args:
        model: HGT model
        graph: Full graph data
        loader: Data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        neg_samples: Number of negative samples per positive
    
    Returns:
        average_loss: Average loss over all batches
    """
    model.train()
    total_loss = 0
    
    # Training loop
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out_dict = model(batch)
        
        # Compute loss for user-place interactions
        loss = 0
        
        # User-place link prediction loss
        if ('user', 'visited', 'place') in batch.edge_index_dict:
            edge_index = batch['user', 'visited', 'place'].edge_index
            src, dst = edge_index
            pos_score = model.predict_user_place(
                out_dict['user'][src], 
                out_dict['place'][dst]
            )
            
            # Generate negative edges
            neg_src, neg_dst = create_negative_edges(
                edge_index, 
                batch['user'].num_nodes,
                batch['place'].num_nodes,
                num_samples=neg_samples
            )
            
            neg_score = model.predict_user_place(
                out_dict['user'][neg_src], 
                out_dict['place'][neg_dst]
            )
            
            # Binary cross-entropy for link prediction
            pos_loss = -torch.log(pos_score + 1e-15).mean()
            neg_loss = -torch.log(1 - neg_score + 1e-15).mean()
            
            up_loss = pos_loss + neg_loss
            loss += up_loss
        
        # User-user link prediction loss
        if ('user', 'friends_with', 'user') in batch.edge_index_dict:
            edge_index = batch['user', 'friends_with', 'user'].edge_index
            src, dst = edge_index
            pos_score = model.predict_user_user(
                out_dict['user'][src], 
                out_dict['user'][dst]
            )
            
            # Generate negative edges
            neg_src, neg_dst = create_negative_edges(
                edge_index, 
                batch['user'].num_nodes,
                batch['user'].num_nodes,
                num_samples=neg_samples
            )
            
            neg_score = model.predict_user_user(
                out_dict['user'][neg_src], 
                out_dict['user'][neg_dst]
            )
            
            # Binary cross-entropy for link prediction
            pos_loss = -torch.log(pos_score + 1e-15).mean()
            neg_loss = -torch.log(1 - neg_score + 1e-15).mean()
            
            uu_loss = pos_loss + neg_loss
            loss += uu_loss
        
        # Compute gradients and update parameters
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    return avg_loss


def main():
    """Main training function."""
    args = parse_args()
    config = load_config(args.config)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup random seeds for reproducibility
    setup_seeds(config.get('seed', 42))
    
    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Load graph data
    print("Loading graph data...")
    graph, metadata = load_graph(config['data'])
    
    # Create model
    in_channels_dict = {
        'user': graph['user'].x.size(1),
        'place': graph['place'].x.size(1),
        'event': graph['event'].x.size(1) if 'event' in graph.node_types else 0,
        'group': graph['group'].x.size(1) if 'group' in graph.node_types else 0
    }
    
    print("Creating model...")
    model = create_model(config['model'], metadata, in_channels_dict)
    model = model.to(device)
    
    # Create data loaders
    if config['training'].get('use_hgt_loader', True):
        loader = HGTLoader(
            graph,
            num_samples={key: [10, 5] for key in metadata[1]},
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training'].get('num_workers', 4)
        )
    else:
        loader = NeighborLoader(
            graph,
            num_neighbors=[10, 5],
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training'].get('num_workers', 4)
        )
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=config['training'].get('patience', 5),
        verbose=True
    )
    
    # Create loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Setup TensorBoard
    writer = SummaryWriter(args.log_dir)
    
    # Training loop
    best_val_auc = 0
    for epoch in range(config['training']['num_epochs']):
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train for one epoch
        start_time = time.time()
        train_loss = train_epoch(
            model, graph, loader, optimizer, criterion, device,
            neg_samples=config['training'].get('neg_samples', 1)
        )
        train_time = time.time() - start_time
        
        # Evaluate on validation set
        val_metrics = evaluate_model(
            model, graph, device,
            task_types=['user_place', 'user_user']
        )
        
        # Log metrics
        print(f"Train Loss: {train_loss:.4f}, Time: {train_time:.2f}s")
        for task, metrics in val_metrics.items():
            print(f"Validation {task} - AUC: {metrics['auc']:.4f}, AP: {metrics['ap']:.4f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        for task, metrics in val_metrics.items():
            writer.add_scalar(f'AUC/{task}', metrics['auc'], epoch)
            writer.add_scalar(f'AP/{task}', metrics['ap'], epoch)
        
        # Update learning rate
        avg_auc = np.mean([metrics['auc'] for metrics in val_metrics.values()])
        scheduler.step(train_loss)
        
        # Save model if it's the best so far
        if avg_auc > best_val_auc:
            best_val_auc = avg_auc
            model_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, model_path)
            print(f"Model saved to {model_path}")
        
        # Save checkpoint
        if (epoch + 1) % config['training'].get('checkpoint_interval', 10) == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_metrics': val_metrics,
                'config': config
            }, checkpoint_path)
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save({
        'epoch': config['training']['num_epochs'] - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, final_path)
    print(f"Final model saved to {final_path}")
    
    # Close TensorBoard writer
    writer.close()


if __name__ == '__main__':
    main()