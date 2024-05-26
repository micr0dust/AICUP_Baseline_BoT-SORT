import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('font', family='Yu Gothic')

def plot_bar_chart(data, labels):
    # Create a list of x-coordinates for the bars
    x = range(len(data))
    
    # Plot the bars
    plt.bar(x, data)
    
    # Add labels and title
    plt.xlabel('最大停留貞數', size=20)
    plt.ylabel('車輛數', size=20)
    plt.title('不同車輛在監視器內停留貞數', size=20)
    
    # Set the labels for the x ticks
    plt.xticks(x, labels)  # Add fontproperties here
    
    # Show the plot
    plt.show()

# Example usage
data = [0, 1224, 1025, 667, 489, 393, 275, 213, 142, 112, 510]
plot_bar_chart(data, list(range(len(data))))
