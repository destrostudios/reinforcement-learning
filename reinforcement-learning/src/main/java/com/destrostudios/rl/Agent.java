package com.destrostudios.rl;

import ai.djl.ndarray.NDList;

public interface Agent {

    NDList chooseAction(Environment environment);

    void train(Replay[] replays);
}
