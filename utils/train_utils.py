
import matplotlib.pyplot as plt
import numpy as np

def plot_predict(train_x, train_y, train_pred,
                    valid_x, valid_y, valid_pred, plot_numbers = 10):
    for i in range(plot_numbers):
        i = i*11
        plt.figure(figsize = (15, 3))
        
        plt.subplot(1,2,1)
        import numpy as np
        plt.plot(np.arange(20), train_x[i,:],   # m_train_time.cpu().detach().numpy()[i,:,-2],
                marker = 'o', color = 'black', label = 'True_input')
        plt.plot(np.arange(21,28), train_y[i],    #m_train_y[:,:,0].cpu().detach().numpy()[i],
                marker = 'o', color = 'red', label = 'True_output', alpha = 0.5)
        plt.plot(np.arange(21,28),train_pred[i],    #.cpu().detach().numpy()[i],
                color = 'blue', label = 'Predict', marker = 'x', ls = '--', alpha = 0.5)
        plt.title('train')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(np.arange(20), valid_x[i,:],
                marker = 'o', color = 'black', label = 'True_input')
        plt.plot(np.arange(21,28), valid_y[i],
                marker = 'o', color = 'red', label = 'True_output', alpha = 0.5)
        plt.plot(np.arange(21,28), valid_pred[i],
                color = 'blue', label = 'Predict', marker = 'x', ls = '--', alpha = 0.5)
        plt.title('validation')
        plt.legend()
        plt.show()
