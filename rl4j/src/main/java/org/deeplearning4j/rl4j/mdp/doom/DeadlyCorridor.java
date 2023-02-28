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

package org.deeplearning4j.rl4j.mdp.doom;

import java.util.Arrays;
import java.util.List;

import org.deeplearning4j.rl4j.mdp.doom.vizdoom.Button;

public class DeadlyCorridor extends Doom {

    public DeadlyCorridor(boolean render) {
        super(render);
    }

    public Configuration getConfiguration() {
        setScaleFactor(1.0);
        List<Button> buttons = Arrays.asList(Button.ATTACK, Button.MOVE_LEFT, Button.MOVE_RIGHT, Button.MOVE_FORWARD,
                        Button.TURN_LEFT, Button.TURN_RIGHT);



        return new Configuration("deadly_corridor", 0.0, 5, 100, 2100, 0, buttons);
    }
}

