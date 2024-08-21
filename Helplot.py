import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.style.use("seaborn-whitegrid")

class Helplot:
  def __init__(self, history, hist_acc="accuracy"):
    self.h = history
    self.l = len(history.history['loss']) + 1
    self.e = [*range(1, self.l)]
    self.hist_acc = hist_acc
  
  @property
  def trainplot(self):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(self.e, self.h.history[self.hist_acc], marker="x", c="b")
    ax[0].set_title("Train Point")
    ax[0].set_xticks(self.e)
    ax[1].plot(self.e, self.h.history["loss"], marker="x", c="r")
    ax[1].set_title("Train Loss")
    ax[1].set_xticks(self.e)
    fig.suptitle("Train Data Plot")
    plt.tight_layout();

  @property
  def testiplot(self):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(self.e, self.h.history[f"val_{self.hist_acc}"], marker="x", c="b")
    ax[0].set_title("Test Point")
    ax[0].set_xticks(self.e)
    ax[1].plot(self.e, self.h.history["val_loss"], marker="x", c="r")
    ax[1].set_title("Test Loss")
    ax[1].set_xticks(self.e)
    fig.suptitle("Test Data Plot")
    plt.tight_layout();