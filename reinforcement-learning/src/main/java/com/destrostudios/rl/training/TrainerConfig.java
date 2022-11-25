package com.destrostudios.rl.training;

import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.TrainingConfig;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.initializer.NormalInitializer;
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

    @Builder.Default
    private int outcomeBatchSize = 32;
    @Builder.Default
    private int outcomeBufferSize = 50000; // Number of previous outcomes to remember
    @Builder.Default
    private float rewardDiscount = 0.9f; // Decay rate of past observations
    @Builder.Default
    private float initialEpsilon = 0.01f;
    @Builder.Default
    private float finalEpsilon = 0.0001f;
    @Builder.Default
    private int environmentStepsObserve = 1000; // Environment steps to observe before training
    @Builder.Default
    private int trainStepsExplore = 3000000; // Train steps over which to anneal epsilon
    @Builder.Default
    private int trainStepsSaveInterval = 100000; // Save model every x train steps
    @Builder.Default
    private TrainingConfig trainingConfig = getDefaultTrainingConfig();

    public static DefaultTrainingConfig getDefaultTrainingConfig() {
        return new DefaultTrainingConfig(Loss.l2Loss())
                .optOptimizer(Adam.builder().optLearningRateTracker(Tracker.fixed(1e-6f)).build())
                .addEvaluator(new Accuracy())
                .optInitializer(new NormalInitializer())
                .addTrainingListeners(TrainingListener.Defaults.basic());
    }
}
