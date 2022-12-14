package com.destrostudios.rl.util;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;

import java.util.ArrayDeque;
import java.util.Queue;

public class NDContinuousArray {

    public NDContinuousArray(int count) {
        this.count = count;
        queue = new ArrayDeque<>(count);
    }
    private int count;
    private Queue<NDArray> queue;

    /**
     * Copy the initial array to all positions, then shift and replace the last position each frame to ensure that the batch is continuous.
     */
    public NDList push(NDArray array) {
        if (queue.isEmpty()) {
            for (int i = 0; i < count; i++) {
                queue.offer(array);
            }
        } else {
            queue.remove();
            queue.offer(array);
        }
        return new NDList(NDArrays.stack(new NDList(queue), 1));
    }
}
