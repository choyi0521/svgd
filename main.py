from experiments import ToyExample, BNNExperiment


if __name__ == '__main__':
    te=ToyExample()
    te.process()

    bnn = BNNExperiment()
    bnn.train('RMSE')
    #bnn.train('LL')
