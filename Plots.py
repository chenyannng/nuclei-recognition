import matplotlib.pyplot as plt

class Plots():

    def __init__(self, train_history):
        self.history = train_history.history

    def plot_metrics(self, name_accuracy='sparse_categorical_accuracy', name_loss='loss'):
        loss = self.history[name_loss]
        accuracy = self.history[name_accuracy]
        
        #plot accuracy and loss
        epoch_id = list(range(len(loss)))
        plt.subplot(121)
        plt.plot(epoch_id, accuracy)
        plt.ylabel(name_accuracy)
        plt.xlabel('Epoch')
        plt.subplot(122)
        plt.plot(epoch_id, loss)
        plt.ylabel(name_loss)
        plt.xlabel('Epoch')
        plt.show()
        