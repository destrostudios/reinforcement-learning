package com.destrostudios.rl;

import ai.djl.ndarray.NDManager;
import lombok.AllArgsConstructor;
import lombok.Getter;

@AllArgsConstructor
@Getter
public class Outcome {

    private NDManager manager;
    private float reward;
    private boolean terminal;
}
