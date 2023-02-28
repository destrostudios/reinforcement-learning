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
package org.deeplearning4j.rl4j.environment;

import org.nd4j.linalg.api.rng.Random;

public class ValidIntegerActionSchema extends IntegerActionSchema {

    public ValidIntegerActionSchema(int numActions, int noOpAction, Environment<Integer> environment) {
        super(numActions, noOpAction);
        this.environment = environment;
    }

    public ValidIntegerActionSchema(int numActions, int noOpAction, Random rnd, Environment<Integer> environment) {
        super(numActions, noOpAction, rnd);
        this.environment = environment;
    }
    private Environment<Integer> environment;

    @Override
    public Integer getRandomAction() {
        // Good enough for now
        int action;
        do {
            action = rnd.nextInt(actionSpaceSize);
        } while (!environment.isValidAction(action));
        return action;
    }
}
