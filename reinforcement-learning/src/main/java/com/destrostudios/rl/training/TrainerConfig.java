package com.destrostudios.rl.training;

import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.TrainingConfig;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.tracker.Tracker;
import lombok.*;

@NoArgsConstructor
@AllArgsConstructor(access = AccessLevel.PRIVATE)
@Builder
@Setter
@Getter
public class TrainerConfig {

    private Shape[] shapes;
    @Builder.Default
    private int replayBatchSize = 32;
    @Builder.Default
    private int replayBufferSize = 50000; // Number of previous outcomes to remember
    @Builder.Default
    private float rewardDiscount = 0.9f; // Decay rate of past observations
    @Builder.Default
    private float initialEpsilon = 1;
    @Builder.Default
    private float finalEpsilon = 0.001f;
    @Builder.Default
    private int environmentStepsExplore = 50000; // Train steps over which to anneal epsilon
    @Builder.Default
    private int trainStepsTotal = 3000000;
    @Builder.Default
    private int trainStepsSaveInterval = 100000; // Save model every x train steps
    @Builder.Default
    private TrainingConfig trainingConfig = getDefaultTrainingConfig();

    public static DefaultTrainingConfig getDefaultTrainingConfig() {
        return new DefaultTrainingConfig(Loss.l2Loss())
                .optOptimizer(Adam.builder().optLearningRateTracker(Tracker.fixed(0.00001f)).build())
                //.addEvaluator(new Accuracy())
                //.optInitializer(new NormalInitializer())
                .addTrainingListeners(TrainingListener.Defaults.basic());
    }
}
