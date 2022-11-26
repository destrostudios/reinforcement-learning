package com.destrostudios.rl.test;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import com.destrostudios.rl.Environment;
import com.destrostudios.rl.test.game.Constant;
import com.destrostudios.rl.training.Trainer;
import com.destrostudios.rl.test.game.FlappyBird;
import com.destrostudios.rl.training.TrainerConfig;

import java.io.IOException;

public class TestTrainer {

    public static void main(String[] args) throws IOException, MalformedModelException {
        Environment environment = new FlappyBird();
        Model model = TestModelLoader.loadModel();
        Trainer trainer = new Trainer(environment, TrainerConfig.builder()
                .shape(Constant.OBSERVATION_SHAPE)
                .build());
        trainer.train(model);
    }
}
