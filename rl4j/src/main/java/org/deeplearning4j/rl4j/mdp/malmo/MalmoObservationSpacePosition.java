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

package org.deeplearning4j.rl4j.mdp.malmo;

import org.json.JSONObject;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.microsoft.msr.malmo.TimestampedStringVector;
import com.microsoft.msr.malmo.WorldState;

public class MalmoObservationSpacePosition extends MalmoObservationSpace {
    @Override
    public String getName() {
        return "Box(5,)";
    }

    @Override
    public int[] getShape() {
        return new int[] {5};
    }

    @Override
    public INDArray getLow() {
        INDArray low = Nd4j.create(new float[] {Integer.MIN_VALUE, Integer.MIN_VALUE, Integer.MIN_VALUE});
        return low;
    }

    @Override
    public INDArray getHigh() {
        INDArray high = Nd4j.create(new float[] {Integer.MAX_VALUE, Integer.MAX_VALUE, Integer.MAX_VALUE});
        return high;
    }

    @Override
    public double[] decode(WorldState world_state) {
    	TimestampedStringVector observations = world_state.getObservations();

        if (observations.isEmpty())
            return null;

        String obs_text = observations.get((int) (observations.size() - 1)).getText();

        JSONObject observation = new JSONObject(obs_text);

        double xpos = observation.getDouble("XPos");
        double ypos = observation.getDouble("YPos");
        double zpos = observation.getDouble("ZPos");
        double yaw = observation.getDouble("Yaw");
        double pitch = observation.getDouble("Pitch");

        return new double[] {xpos, ypos, zpos, yaw, pitch};
    }
}
