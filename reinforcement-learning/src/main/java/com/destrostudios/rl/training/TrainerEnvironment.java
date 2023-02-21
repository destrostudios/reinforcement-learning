package com.destrostudios.rl.training;

import ai.djl.modality.rl.ActionSpace;
import ai.djl.modality.rl.LruReplayBuffer;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import com.destrostudios.rl.Environment;
import lombok.AllArgsConstructor;
import lombok.Setter;

@AllArgsConstructor
public class TrainerEnvironment<Observation> implements RlEnv {

    public TrainerEnvironment(NDManager baseManager, Environment<Observation> environment, TrainerConfig config) {
        this.environment = environment;
        this.baseManager = baseManager;
        replayBuffer = new LruReplayBuffer(config.getReplayBatchSize(), config.getReplayBufferSize());
        environment.initialize(baseManager);
    }
    private Environment<Observation> environment;
    private LruReplayBuffer replayBuffer;
    private NDManager baseManager;
    @Setter
    private NDManager trainSubManager;

    @Override
    public void reset() {
        environment.reset();
    }

    @Override
    public NDList getObservation() {
        return environment.mapObservation(trainSubManager, environment.getObservation());
    }

    @Override
    public ActionSpace getActionSpace() {
        return environment.getActionSpace();
    }

    @Override
    public Step step(NDList action, boolean training) {
        NDManager stepSubManager = baseManager.newSubManager();
        NDList preObservation = environment.mapObservation(stepSubManager, environment.getObservation());
        preObservation.attach(stepSubManager);
        float reward = environment.takeAction(action);
        NDList postObservation = environment.mapObservation(stepSubManager, environment.getObservation());
        postObservation.attach(stepSubManager);
        TrainerEnvironmentStep step = new TrainerEnvironmentStep(stepSubManager, preObservation, action, postObservation, environment.getActionSpace(), stepSubManager.create(reward), environment.isTerminated());
        if (training && (step.getReward().getFloat() != 0)) {
            replayBuffer.addStep(step);
        }
        return step;
    }

    @Override
    public Step[] getBatch() {
        return replayBuffer.getBatch();
    }

    @Override
    public void close() {
        baseManager.close();
    }
}
