package com.destrostudios.rl.test;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import com.destrostudios.rl.test.game.Constant;

import java.io.IOException;
import java.nio.file.Paths;

public class TestModelLoader {

    private static final String MODEL_PATH = Constant.RESOURCE_PATH + "/model";

    public static Model loadModel() throws IOException, MalformedModelException {
        Model model = Model.newInstance("QNetwork");
        model.setBlock(getBlock());
        // model.load(Paths.get(MODEL_PATH), "dqn-trained");
        return model;
    }

    private static SequentialBlock getBlock() {
        return new SequentialBlock()
            .add(arrays -> {
                NDArray observation = arrays.get(0); // Shape(N, 6)
                NDArray action = arrays.get(1).toType(DataType.FLOAT32, true); // Shape(N, 2)

                // Concatenate to a combined vector of Shape(N, 8)
                NDArray combined = NDArrays.concat(new NDList(observation, action), 1);

                return new NDList(combined);
            })
            .add(new Mlp(8, 1, new int[]{ 512, 512 }));
    }
}
