package com.destrostudios.rl.test;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import com.destrostudios.rl.training.Trainer;
import com.destrostudios.rl.test.game.FlappyBird;

import java.io.IOException;

public class TestTrainer {

    public static void main(String[] args) throws IOException, MalformedModelException {
        int batchSize = 32;
        FlappyBird environment = new FlappyBird(NDManager.newBaseManager(), batchSize, Trainer.REPLAY_BUFFER_SIZE, true);
        Model model = TestModelLoader.loadModel();
        Trainer trainer = new Trainer(environment, Trainer.createDefaultConfig(), batchSize);
        trainer.train(model);
    }
}
