import torch

def batch_train(
    model,
    train_loader,
    optimizer,
    loss_fn,
    scheduler,
    device
    ):
    train_loss = 0
    num_molecules = 0
    model.train()
    for batch in train_loader:
        batch = batch.to(device)

        x, edge_index, batch_seg = batch.x, \
        batch.edge_index, batch.batch

        optimizer.zero_grad()
        pred = model(x, edge_index, batch_seg)
        loss = loss_fn(pred.view(-1, 1), 
                        batch.spectrum.view(-1, 1))
        loss.backward()
        train_loss += loss
        num_molecules += batch.num_graphs
        optimizer.step()
    scheduler.step()
    
    train_avg_loss = train_loss / num_molecules
    return train_avg_loss

@torch.no_grad()
def batch_val(
    model,
    val_loader,
    loss_fn,
    device
    ):
    val_epoch_loss = 0
    num_molecules = 0
    model.eval()   
    for batch in val_loader:
        batch = batch.to(device)

        x, edge_index, batch_seg = batch.x, \
        batch.edge_index, batch.batch
        pred = model(x, edge_index, batch_seg)
        loss = loss_fn(pred.view(-1, 1), 
                        batch.spectrum.view(-1, 1))
        val_epoch_loss += loss
        num_molecules += batch.num_graphs

    val_avg_loss = val_epoch_loss / num_molecules

    return val_avg_loss