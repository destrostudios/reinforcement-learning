/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.rl4j.mdp;

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;

import java.util.Random;

public class CartpoleNative implements MDP<Box, Integer, DiscreteSpace> {
    public enum KinematicsIntegrators { Euler, SemiImplicitEuler }

    private static final int NUM_ACTIONS = 2;
    private static final int ACTION_LEFT = 0;
    private static final int ACTION_RIGHT = 1;
    private static final int OBSERVATION_NUM_FEATURES = 4;

    private static final double gravity = 9.8;
    private static final double massCart = 1.0;
    private static final double massPole = 0.1;
    private static final double totalMass = massPole + massCart;
    private static final double length = 0.5; // actually half the pole's length
    private static final double polemassLength = massPole * length;
    private static final double forceMag = 10.0;
    private static final double tau = 0.02;  // seconds between state updates

    // Angle at which to fail the episode
    private static final double thetaThresholdRadians = 12.0 * 2.0 * Math.PI / 360.0;
    private static final double xThreshold = 2.4;

    private final Random rnd;

    @Getter @Setter
    private KinematicsIntegrators kinematicsIntegrator = KinematicsIntegrators.Euler;

    @Getter
    private boolean done = false;

    private double x;
    private double xDot;
    private double theta;
    private double thetaDot;
    private Integer stepsBeyondDone;

    @Getter
    private DiscreteSpace actionSpace = new DiscreteSpace(NUM_ACTIONS);
    @Getter
    private ObservationSpace<Box> observationSpace = new ArrayObservationSpace(new int[] { OBSERVATION_NUM_FEATURES });

    public CartpoleNative() {
        rnd = new Random();
    }

    public CartpoleNative(int seed) {
        rnd = new Random(seed);
    }

    @Override
    public Box reset() {

        x = 0.1 * rnd.nextDouble() - 0.05;
        xDot = 0.1 * rnd.nextDouble() - 0.05;
        theta = 0.1 * rnd.nextDouble() - 0.05;
        thetaDot = 0.1 * rnd.nextDouble() - 0.05;
        stepsBeyondDone = null;
        done = false;

        return new Box(x, xDot, theta, thetaDot);
    }

    @Override
    public void close() {

    }

    @Override
    public StepReply<Box> step(Integer action) {
        double force = action == ACTION_RIGHT ? forceMag : -forceMag;
        double cosTheta = Math.cos(theta);
        double sinTheta = Math.sin(theta);
        double temp = (force + polemassLength * thetaDot * thetaDot * sinTheta) / totalMass;
        double thetaAcc = (gravity * sinTheta - cosTheta* temp) / (length * (4.0/3.0 - massPole * cosTheta * cosTheta / totalMass));
        double xAcc = temp - polemassLength * thetaAcc * cosTheta / totalMass;

        switch(kinematicsIntegrator) {
            case Euler:
                x += tau * xDot;
                xDot += tau * xAcc;
                theta += tau * thetaDot;
                thetaDot += tau * thetaAcc;
                break;

            case SemiImplicitEuler:
                xDot += tau * xAcc;
                x += tau * xDot;
                thetaDot += tau * thetaAcc;
                theta += tau * thetaDot;
                break;
        }

        done |=  x < -xThreshold || x > xThreshold
                || theta < -thetaThresholdRadians || theta > thetaThresholdRadians;

        double reward;
        if(!done) {
            reward = 1.0;
        }
        else if(stepsBeyondDone == null) {
            stepsBeyondDone = 0;
            reward = 1.0;
        }
        else {
            ++stepsBeyondDone;
            reward = 0;
        }

        return new StepReply<>(new Box(x, xDot, theta, thetaDot), reward, done, null);
    }

    @Override
    public MDP<Box, Integer, DiscreteSpace> newInstance() {
        return new CartpoleNative();
    }

}
