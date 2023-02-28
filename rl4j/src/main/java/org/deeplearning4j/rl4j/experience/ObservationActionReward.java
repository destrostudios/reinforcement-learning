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
package org.deeplearning4j.rl4j.experience;

import lombok.AllArgsConstructor;
import lombok.Getter;

import org.deeplearning4j.rl4j.environment.action.Action;
import org.deeplearning4j.rl4j.environment.observation.Observation;
import org.deeplearning4j.rl4j.environment.observation.ObservationSource;

@AllArgsConstructor
public class ObservationActionReward<OBSERVATION extends Observation, ACTION extends Action> implements ObservationSource {

    /**
     * The observation before the action is taken
     */
    @Getter
    private final OBSERVATION observation;

    @Getter
    private final ACTION action;

    @Getter
    private final double reward;

    /**
     * True if the episode ended after the action has been taken.
     */
    @Getter
    private final boolean terminal;
}
