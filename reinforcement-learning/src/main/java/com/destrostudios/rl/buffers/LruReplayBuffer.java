package com.destrostudios.rl.buffers;

import ai.djl.util.RandomUtils;
import com.destrostudios.rl.ReplayBuffer;
import com.destrostudios.rl.Replay;

import java.util.ArrayList;

/**
 * A simple {@link ReplayBuffer} that randomly selects across the whole buffer, but always removes the oldest items in the buffer once it is full.
 */
public class LruReplayBuffer implements ReplayBuffer {

    public LruReplayBuffer(int trainBatchSize, int bufferSize) {
        this.trainBatchSize = trainBatchSize;
        replays = new Replay[bufferSize];
        replaysToClose = new ArrayList<>(bufferSize);
        firstReplayIndex = 0;
        replaysActualSize = 0;
    }
    private int trainBatchSize;
    private Replay[] replays;
    private ArrayList<Replay> replaysToClose;
    private int firstReplayIndex;
    private int replaysActualSize;

    @Override
    public void addReplay(Replay replay) {
        if (replaysActualSize == replays.length) {
            int indexToReplace = Math.floorMod(firstReplayIndex - 1, replays.length);
            replaysToClose.add(replays[indexToReplace]);
            replays[indexToReplace] = replay;
            firstReplayIndex = Math.floorMod(firstReplayIndex + 1, replays.length);
        } else {
            replays[replaysActualSize] = replay;
            replaysActualSize++;
        }
    }

    @Override
    public void cleanupInterval() {
        replaysToClose.forEach(Replay::close);
        replaysToClose.clear();
    }

    @Override
    public Replay[] getTrainingBatch() {
        Replay[] batch = new Replay[trainBatchSize];
        for (int i = 0; i < trainBatchSize; i++) {
            int baseIndex = RandomUtils.nextInt(replaysActualSize);
            int index = Math.floorMod(firstReplayIndex + baseIndex, replays.length);
            batch[i] = replays[index];
        }
        return batch;
    }
}
