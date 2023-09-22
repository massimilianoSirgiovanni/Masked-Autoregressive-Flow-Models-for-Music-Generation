from manageFiles import *

def trainModel(tr_set, model, loss_function, optimizer, num_epochs=100):
    # model <-- Modello di ML da allenare
    # loss_function <-- Funzione di perdita
    # optimizer <-- Algoritmo di Ottimizzazione

    #Inserire la divisione in batch

    for epoch in range(num_epochs):
        # Carica dati MIDI e appiattisci (adatta i dati al tuo dataset)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(tr_set)
        loss = loss_function(recon_batch, tr_set, mu, logvar)
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    saveVariableInFile("./variables/model", model)
