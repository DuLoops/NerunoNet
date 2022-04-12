using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNet.Classes;

namespace NeuralNet.Classes
{
    public class Network
    {
        Random random;

        int numLayers;
        int[] sizes;
        double[][,] biases;
        double[][,] weights;

        public Network(int[] layers) {
            random = new Random();

            numLayers = layers.Length;
            sizes = layers;
            InitializeBiases();
            InitializeWeights();
        }

        public Network(byte[] net) {
            int index = 0;
            numLayers = BitConverter.ToInt32(net, index);
            index += 4;

            sizes = new int[numLayers];
            for (int i = 0; i < numLayers; ++i) {
                sizes[i] = BitConverter.ToInt32(net, index);
                index += 4;
            }

            biases = new double[numLayers - 1][,];
            for (int y = 1; y < numLayers; ++y) {
                int layer = sizes[y];
                biases[y - 1] = ReadDoubles(layer, 1, net, ref index);
            }

            weights = new double[numLayers - 1][,];
            for (int i = 0; i < numLayers - 1; ++i) {
                int x = sizes[i];
                int y = sizes[i + 1];
                weights[i] = ReadDoubles(y, x, net, ref index);
            }
        }

        private double[,] ReadDoubles(int d1, int d2, byte[] data, ref int index) {
            double[,] result = new double[d1, d2];

            for (int i = 0; i < d1; ++i) {
                for (int j = 0; j < d2; ++j) {
                    result[i, j] = BitConverter.ToDouble(data, index);
                    index += 8;
                }
            }

            return result;
        }

        public byte[] GetNetworkAsBytes() {
            List<byte> bytes = new List<byte>();

            bytes.AddRange(BitConverter.GetBytes(numLayers));

            for (int i = 0; i < sizes.Length; ++i) {
                bytes.AddRange(BitConverter.GetBytes(sizes[i]));
            }

            for (int i = 0; i < biases.Length; ++i) {
                for (int x = 0; x < biases[i].GetLength(0); ++x) {
                    for (int y = 0; y < biases[i].GetLength(1); ++y) {
                        bytes.AddRange(BitConverter.GetBytes(biases[i][x, y]));
                    }
                }
            }

            for (int i = 0; i < weights.Length; ++i) {
                for (int x = 0; x < weights[i].GetLength(0); ++x) {
                    for (int y = 0; y < weights[i].GetLength(1); ++y) {
                        bytes.AddRange(BitConverter.GetBytes(weights[i][x, y]));
                    }
                }
            }

            return bytes.ToArray();
        }

        private void InitializeBiases() {
            biases = new double[numLayers - 1][,];

            for (int y = 1; y < numLayers; ++y) {
                int layer = sizes[y];

                biases[y - 1] = RandomN(layer, 1);
            }
        }

        private void InitializeWeights() {
            weights = new double[numLayers - 1][,];

            for (int i = 0; i < numLayers - 1; ++i) {
                int x = sizes[i];
                int y = sizes[i + 1];

                weights[i] = RandomN(y, x);
            }
        }

        private double[,] RandomN(int d1, int d2) {
            double[,] result = new double[d1, d2];

            for (int i = 0; i < d1; ++i) {
                for (int j = 0; j < d2; ++j) {
                    double u1 = 1.0f - random.NextDouble();
                    double u2 = 1.0f - random.NextDouble();

                    // This is the Box-Muller transform that should create a normal distribution
                    double randStdNormal = Math.Sqrt(-2.0f * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);

                    result[i, j] = randStdNormal;
                }
            }

            return result;
        }

        public double[,] FeedForward(double[,] a) {
            for (int i = 0; i < biases.Length; ++i) {
                var b = biases[i];
                var w = weights[i];

                a = NNHelper.Sigmoid(w.MatMul(a).MatAdd(b));
            }

            return a;
        }

        public void SGD(Tuple<double[,], byte>[] trainingData, int epochs, int miniBatchSize, double eta, Tuple<double[,], byte>[]? testData = null) {
            int? nTest = null;
            if (testData != null) nTest = testData.Length;

            int n = trainingData.Length;

            for (int j = 0; j < epochs; ++j) {
                random.Shuffle(trainingData);

                List<Tuple<double[,], byte>[]> miniBatches = new List<Tuple<double[,], byte>[]>();

                for (int k = 0; k < n; k += miniBatchSize) {
                    var miniBatch = new Tuple<double[,], byte>[miniBatchSize];
                    Array.Copy(trainingData, k, miniBatch, 0, miniBatchSize);
                    miniBatches.Add(miniBatch);
                }

                int batchNum = 1;
                foreach (var miniBatch in miniBatches) {
                    UpdateMiniBatch(miniBatch, eta);
                    //Trace.WriteLine($"Batch done: {batchNum++} / {miniBatches.Count}");
                }

                if (testData != null) {
                    Trace.WriteLine($"Epoch {j}: {Evaluate(testData)} / {nTest}");
                } else {
                    Trace.WriteLine($"Epoch {j} complete");
                }
            }
        }

        public void UpdateMiniBatch(Tuple<double[,], byte>[] miniBatch, double eta) {

            double[][,] nablaB = new double[biases.GetLength(0)][,];
            double[][,] nablaW = new double[weights.GetLength(0)][,];

            for (int i = 0; i < biases.Length; ++i) {
                nablaB[i] = new double[biases[i].GetLength(0), biases[i].GetLength(1)];
                nablaW[i] = new double[weights[i].GetLength(0), weights[i].GetLength(1)];
            }

            for (int x = 0; x < miniBatch.Length; ++x) {
                byte y = miniBatch[x].Item2;

                var (deltaNablaB, deltaNablaW) = Backprop(miniBatch[x].Item1, y);

                for (int j = 0; j < nablaB.Length; ++j) {
                    nablaB[j] = nablaB[j].MatAdd(deltaNablaB[j]);
                    nablaW[j] = nablaW[j].MatAdd(deltaNablaW[j]);
                }
            }

            for (int i = 0; i < weights.Length; ++i) {
                weights[i] = weights[i].MatSub(nablaW[i].MatSca(eta / miniBatch.Length));
                biases[i] = biases[i].MatSub(nablaB[i].MatSca(eta / miniBatch.Length));
            }
        }

        public (double[][,], double[][,]) Backprop(double[,] x, byte y) {
            double[][,] nablaB = new double[biases.Length][,];
            double[][,] nablaW = new double[weights.Length][,]; 

            for (int i = 0; i < biases.Length; ++i) {
                nablaB[i] = new double[biases[i].GetLength(0), biases[i].GetLength(1)];
                nablaW[i] = new double[weights[i].GetLength(0), weights[i].GetLength(1)];
            }

            double[,] activation = x;

            List<double[,]> activations = new List<double[,]>();
            activations.Add(activation);

            List<double[,]> zs = new List<double[,]>();

            for (int i = 0; i < biases.Length; ++i) {
                var b = biases[i];
                var w = weights[i];

                var z = b.MatAdd(w.MatMul(activation));
                zs.Add(z);

                activation = NNHelper.Sigmoid(z);
                activations.Add(activation);
            }

            var delta = CostDerivative(activations[activations.Count - 1], y).MatComMul(NNHelper.SigmoidPrime(zs[zs.Count - 1]));

            nablaB[nablaB.GetLength(0) - 1] = delta;
            nablaW[nablaW.GetLength(0) - 1] = delta.MatMul(activations[activations.Count - 2].Transpose());

            for (int l = 2; l < numLayers; ++l) {
                var z = zs[zs.Count - l];
                var sp = NNHelper.SigmoidPrime(z);
                delta = weights[weights.GetLength(0) - l + 1].Transpose().MatMul(delta).MatComMul(sp);

                nablaB[nablaB.GetLength(0) - l] = delta;
                nablaW[nablaW.GetLength(0) - l] = delta.MatMul(activations[activations.Count - l - 1].Transpose());
            }

            return (nablaB, nablaW);
        }

        public int Evaluate(Tuple<double[,], byte>[] testData) {
            

            var testResults = new Tuple<int, byte>[testData.Length];

            for (int i = 0; i < testData.Length; ++i) {
                double[,] feedResult = FeedForward(testData[i].Item1);
                int foundLayer = NNHelper.ArgMax(feedResult);

                testResults[i] = Tuple.Create(foundLayer, testData[i].Item2);
            }

            int sum = 0;
            for (int i = 0; i < testResults.Length; ++i) {
                if (testResults[i].Item1 == testResults[i].Item2) {
                    //Trace.WriteLine(testResults[i].Item2);
                    ++sum;
                }
            }

            return sum;
        }

        public int CustomTest(double[] testData) {
            double[,] temp = new double[testData.Length, 1];

            for (int i = 0; i < testData.Length; ++i) {
                temp[i, 0] = testData[i];
            }

            double[,] feedResult = FeedForward(temp);
            int foundLayer = NNHelper.ArgMax(feedResult);

            return foundLayer;
        }

        public double[,] CostDerivative(double[,] outputActivations, byte y) {
            double[] arr = new double[10];
            arr[y] = 1;

            return outputActivations.MatSub(arr);
        }

    }
}
