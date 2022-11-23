package com.destrostudios.rl.buffers;

import ai.djl.util.RandomUtils;
import com.destrostudios.rl.EnvironmentStep;
import com.destrostudios.rl.ReplayBuffer;

import java.util.ArrayList;

/**
 * A simple {@link ReplayBuffer} that randomly selects across the whole buffer, but always removes the oldest items in the buffer once it is full.
 */
public class LruReplayBuffer implements ReplayBuffer {

    public LruReplayBuffer(int trainBatchSize, int bufferSize) {
        this.trainBatchSize = trainBatchSize;
        steps = new EnvironmentStep[bufferSize];
        stepToClose = new ArrayList<>(bufferSize);
        firstStepIndex = 0;
        stepsActualSize = 0;
    }
    private int trainBatchSize;
    private EnvironmentStep[] steps;
    private ArrayList<EnvironmentStep> stepToClose;
    private int firstStepIndex;
    private int stepsActualSize;

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

    @Override
    public void cleanupInterval() {
        for (EnvironmentStep step : stepToClose) {
            step.close();
        }
        stepToClose.clear();
    }

    @Override
    public EnvironmentStep[] getTrainingBatch() {
        EnvironmentStep[] batch = new EnvironmentStep[trainBatchSize];
        for (int i = 0; i < trainBatchSize; i++) {
            int baseIndex = RandomUtils.nextInt(stepsActualSize);
            int index = Math.floorMod(firstStepIndex + baseIndex, steps.length);
            batch[i] = steps[index];
        }
        return batch;
    }
}
