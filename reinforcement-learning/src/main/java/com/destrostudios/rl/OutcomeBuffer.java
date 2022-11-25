package com.destrostudios.rl;

public interface OutcomeBuffer {

    void addOutcome(Outcome outcome);

    void cleanupInterval();

    Outcome[] getTrainingBatch();
}
