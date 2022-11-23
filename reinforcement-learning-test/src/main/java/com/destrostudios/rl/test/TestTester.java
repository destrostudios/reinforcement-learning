package com.destrostudios.rl.test;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import com.destrostudios.rl.training.Tester;
import com.destrostudios.rl.test.game.FlappyBird;
import com.destrostudios.rl.training.Trainer;

import java.io.IOException;

public class TestTester {

    public static void main(String[] args) throws IOException, MalformedModelException {
        FlappyBird environment = new FlappyBird(NDManager.newBaseManager(), true);
        Model model = TestModelLoader.loadModel();
        Tester tester = new Tester(environment, Trainer.createDefaultConfig());
        tester.test(model);
    }
}
