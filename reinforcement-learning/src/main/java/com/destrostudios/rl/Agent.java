package com.destrostudios.rl;

import ai.djl.ndarray.NDList;

public interface Agent {

    NDList chooseAction(Environment environment, boolean isTraining);

    void train(Replay[] replays);
}
