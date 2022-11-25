package com.destrostudios.rl.buffers;

import ai.djl.util.RandomUtils;
import com.destrostudios.rl.Outcome;
import com.destrostudios.rl.OutcomeBuffer;

import java.util.ArrayList;

/**
 * A simple {@link OutcomeBuffer} that randomly selects across the whole buffer, but always removes the oldest items in the buffer once it is full.
 */
public class LruOutcomeBuffer implements OutcomeBuffer {

    public LruOutcomeBuffer(int trainBatchSize, int bufferSize) {
        this.trainBatchSize = trainBatchSize;
        outcomes = new Outcome[bufferSize];
        outcomesToClose = new ArrayList<>(bufferSize);
        firstOutcomeIndex = 0;
        outcomesActualSize = 0;
    }
    private int trainBatchSize;
    private Outcome[] outcomes;
    private ArrayList<Outcome> outcomesToClose;
    private int firstOutcomeIndex;
    private int outcomesActualSize;

    @Override
    public void addOutcome(Outcome outcome) {
        if (outcomesActualSize == outcomes.length) {
            int indexToReplace = Math.floorMod(firstOutcomeIndex - 1, outcomes.length);
            outcomesToClose.add(outcomes[indexToReplace]);
            outcomes[indexToReplace] = outcome;
            firstOutcomeIndex = Math.floorMod(firstOutcomeIndex + 1, outcomes.length);
        } else {
            outcomes[outcomesActualSize] = outcome;
            outcomesActualSize++;
        }
    }

    @Override
    public void cleanupInterval() {
        for (Outcome step : outcomesToClose) {
            step.close();
        }
        outcomesToClose.clear();
    }

    @Override
    public Outcome[] getTrainingBatch() {
        Outcome[] batch = new Outcome[trainBatchSize];
        for (int i = 0; i < trainBatchSize; i++) {
            int baseIndex = RandomUtils.nextInt(outcomesActualSize);
            int index = Math.floorMod(firstOutcomeIndex + baseIndex, outcomes.length);
            batch[i] = outcomes[index];
        }
        return batch;
    }
}
