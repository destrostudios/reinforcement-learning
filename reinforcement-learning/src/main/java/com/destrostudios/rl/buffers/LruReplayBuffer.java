package com.destrostudios.rl.buffers;

import ai.djl.util.RandomUtils;
import com.destrostudios.rl.EnvironmentStep;
import com.destrostudios.rl.ReplayBuffer;

import java.util.ArrayList;

/**
 * A simple {@link ReplayBuffer} that randomly selects across the whole buffer, but always removes the oldest items in the buffer once it is full.
 */
public class LruReplayBuffer implements ReplayBuffer {

    /**
     * @param batchSize  the number of steps to train on per batch
     * @param bufferSize the number of steps to hold in the buffer
     */
    public LruReplayBuffer(int batchSize, int bufferSize) {
        this.batchSize = batchSize;
        steps = new EnvironmentStep[bufferSize];
        stepToClose = new ArrayList<>(bufferSize);
        firstStepIndex = 0;
        stepsActualSize = 0;
    }
    private int batchSize;
    private EnvironmentStep[] steps;
    private ArrayList<EnvironmentStep> stepToClose;
    private int firstStepIndex;
    private int stepsActualSize;

    @Override
    public EnvironmentStep[] getBatch() {
        EnvironmentStep[] batch = new EnvironmentStep[batchSize];
        for (int i = 0; i < batchSize; i++) {
            int baseIndex = RandomUtils.nextInt(stepsActualSize);
            int index = Math.floorMod(firstStepIndex + baseIndex, steps.length);
            batch[i] = steps[index];
        }
        return batch;
    }

    public void closeStep() {
        for (EnvironmentStep step : stepToClose) {
            step.close();
        }
        stepToClose.clear();
    }

    @Override
    public void addStep(EnvironmentStep step) {
        if (stepsActualSize == steps.length) {
            int stepToReplace = Math.floorMod(firstStepIndex - 1, steps.length);
            stepToClose.add(steps[stepToReplace]);
            steps[stepToReplace] = step;
            firstStepIndex = Math.floorMod(firstStepIndex + 1, steps.length);
        } else {
            steps[stepsActualSize] = step;
            stepsActualSize++;
        }
    }
}
