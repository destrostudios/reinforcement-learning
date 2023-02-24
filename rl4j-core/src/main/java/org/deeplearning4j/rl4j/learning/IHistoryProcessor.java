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

package org.deeplearning4j.rl4j.learning;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface IHistoryProcessor {

    Configuration getConf();

    /** Returns compressed arrays, which must be rescaled based
     *  on the value returned by {@link #getScale()}. */
    INDArray[] getHistory();

    void record(INDArray image);

    void add(INDArray image);

    void startMonitor(String filename, int[] shape);

    void stopMonitor();

    boolean isMonitoring();

    /** Returns the scale of the arrays returned by {@link #getHistory()}, typically 255. */
    double getScale();

    @AllArgsConstructor
    @Builder
    @Data
    public static class Configuration {
        @Builder.Default int historyLength = 4;
        @Builder.Default int rescaledWidth = 84;
        @Builder.Default int rescaledHeight = 84;
        @Builder.Default int croppingWidth = 84;
        @Builder.Default int croppingHeight = 84;
        @Builder.Default int offsetX = 0;
        @Builder.Default int offsetY = 0;
        @Builder.Default int skipFrame = 4;

        public int[] getShape() {
            return new int[] {getHistoryLength(), getRescaledHeight(), getRescaledWidth()};
        }
    }
}
