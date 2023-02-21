package com.destrostudios.rl.training;

import ai.djl.modality.rl.ActionSpace;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import lombok.AllArgsConstructor;
import lombok.Getter;

@AllArgsConstructor
@Getter
public class TrainerEnvironmentStep implements RlEnv.Step {

    private NDManager manager;
    private NDList preObservation;
    private NDList action;
    private NDList postObservation;
    private ActionSpace postActionSpace;
    private NDArray reward;
    private boolean done;

    @Override
    public void close() {
        manager.close();
    }
}
