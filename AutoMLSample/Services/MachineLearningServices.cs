using System.Collections.Immutable;
using System.Data;
using MathNet.Numerics.Statistics;
using AutoMLSample.Models;
using AutoMLSample.Helpers;
using static AutoMLSample.Helpers.ConsoleHelpers;
using Microsoft.ML.AutoML;
using static Microsoft.ML.AutoML.AutoMLExperiment;

namespace AutoMLSample.Services;

internal static class MachineLearningServices
{
    private const string Score = nameof(ModelOutput.Score);
    private const string PredictedLabel = nameof(ModelOutput.PredictedLabel);
    private const string Label = "Label";
    private const string Features = "Features";
    private const string DatasetPath = "Data/sensors_data.csv";

    private static MLContext Context { get; set; } = new MLContext(seed: 1);

    internal static IDataView LoadData()
    {
        var data = Context.Data.LoadFromTextFile<ModelInput>(
            path: DatasetPath,
            hasHeader: true,
            separatorChar: ',');

        var shuffledData = Context.Data.ShuffleRows(data, seed: 1);

        return shuffledData;
    }

    internal static (IDataView trainingDataView, IDataView testingDataView) SplitData(this IDataView data)
    {
        var split = Context.Data.TrainTestSplit(data, testFraction: 0.3);

        return (split.TrainSet, split.TestSet);
    }

    internal static async Task<TrialResult> AutoTrainAsync(uint time, IDataView trainingDataView, IDataView testingDataView)
    {
        WriteLineColor($" INCREASE ML MODEL ACCURACY IN THREE STEPS");
        WriteLineColor($" Learning type: multi-class classification");
        WriteLineColor($" Training time: {time} seconds");
        WriteLineColor("----------------------------------------------------------------------------------");

        Context = new MLContext(seed: 1);

        var trainingDataCollection = Context.Data.CreateEnumerable<ModelInput>(trainingDataView, reuseRowObject: true);
        (var header, _) = trainingDataCollection.ExtractDataAndHeader();

        var preprocessingPipeline = Context.Transforms.Conversion.MapValueToKey(Label)
            .Append(Context.Transforms.Concatenate(Features, header))
            .Append(Context.Transforms.NormalizeMinMax(Features, Features));

        var experimentPipeline = preprocessingPipeline.Append(Context.Auto().MultiClassification());

        var cts = new CancellationTokenSource();
        var experimentSettings = new AutoMLExperimentSettings
        {
            CancellationToken = cts.Token,
            Pipeline = experimentPipeline,
            MaxExperimentTimeInSeconds = time,
            IsMaximizeMetric = false
        };

        var experiment = Context.Auto()
            .CreateExperiment(experimentSettings)
            .SetDataset(trainingDataView, testingDataView)
            .SetEvaluateMetric(MulticlassClassificationMetric.MicroAccuracy, "Label", "PredictedLabel");

        var experimentResult = await experiment.RunAsync();

        return experimentResult;
    }

    internal static ExperimentResult<MulticlassClassificationMetrics> AutoTrain(uint time, IDataView trainingDataView)
    {
        var progressHandler = new MulticlassExperimentProgressHandler();

        var settings = new MulticlassExperimentSettings
        {
            MaxExperimentTimeInSeconds = time,
            OptimizingMetric = MulticlassClassificationMetric.MicroAccuracy,
            CacheDirectoryName = null
        };

        WriteLineColor($" INCREASE ML MODEL ACCURACY IN THREE STEPS");
        WriteLineColor($" Learning type: multi-class classification");
        WriteLineColor($" Training time: {settings.MaxExperimentTimeInSeconds} seconds");
        WriteLineColor("----------------------------------------------------------------------------------");

        Context = new MLContext(seed: 1);

        var experimentResult = Context.Auto()
            .CreateMulticlassClassificationExperiment(settings)
            .Execute(trainingDataView, Label, progressHandler: progressHandler);

        WriteLineColor("----------------------------------------------------------------------------------");
        WriteLineColor($" STEP 1: AutoML experiment result");
        WriteLineColor("----------------------------------------------------------------------------------");
        WriteLineColor($" Top trainers (by accuracy)");
        WriteLineColor("----------------------------------------------------------------------------------");
        PrintTopModels(experimentResult);
        WriteLineColor("----------------------------------------------------------------------------------");
        WriteLineColor($" Selected trainer: {experimentResult.BestRun.TrainerName.Replace("Multi", "")}");
        WriteLineColor("----------------------------------------------------------------------------------");

        return experimentResult;
    }

    internal static void PFI(float threshold, string trainerName, IDataView trainingDataView)
    {
        var trainer = (MulticlassClassificationTrainer)Enum.Parse(typeof(MulticlassClassificationTrainer), trainerName.Replace("Multi", ""));

        var trainingDataCollection = Context.Data.CreateEnumerable<ModelInput>(trainingDataView, reuseRowObject: true);
        (var header, _) = trainingDataCollection.ExtractDataAndHeader();

        var preprocessingPipeline = Context.Transforms.Conversion.MapValueToKey(Label)
            .Append(Context.Transforms.Concatenate(Features, header))
            .Append(Context.Transforms.NormalizeMinMax(Features, Features));

        var dataPrepTransformer = preprocessingPipeline.Fit(trainingDataView);
        var preprocessedTrainData = dataPrepTransformer.Transform(trainingDataView);

        ImmutableArray<MulticlassClassificationMetricsStatistics> pfi;
        dynamic estimator = null;

        estimator = trainer switch
        {
            MulticlassClassificationTrainer.AveragedPerceptronOva => Context.MulticlassClassification.Trainers.OneVersusAll(Context.BinaryClassification.Trainers.AveragedPerceptron()),
            MulticlassClassificationTrainer.FastForestOva => Context.MulticlassClassification.Trainers.OneVersusAll(Context.BinaryClassification.Trainers.FastForest()),
            MulticlassClassificationTrainer.FastTreeOva => Context.MulticlassClassification.Trainers.OneVersusAll(Context.BinaryClassification.Trainers.FastTree()),
            MulticlassClassificationTrainer.LightGbm => Context.MulticlassClassification.Trainers.LightGbm(),
            MulticlassClassificationTrainer.LinearSupportVectorMachinesOva => Context.MulticlassClassification.Trainers.OneVersusAll(Context.BinaryClassification.Trainers.LinearSvm()),
            MulticlassClassificationTrainer.LbfgsMaximumEntropy => Context.MulticlassClassification.Trainers.LbfgsMaximumEntropy(),
            MulticlassClassificationTrainer.LbfgsLogisticRegressionOva => Context.MulticlassClassification.Trainers.OneVersusAll(Context.BinaryClassification.Trainers.LbfgsLogisticRegression()),
            MulticlassClassificationTrainer.SdcaMaximumEntropy => Context.MulticlassClassification.Trainers.SdcaMaximumEntropy(),
            MulticlassClassificationTrainer.SgdCalibratedOva => Context.MulticlassClassification.Trainers.OneVersusAll(Context.BinaryClassification.Trainers.SgdCalibrated()),
            MulticlassClassificationTrainer.SymbolicSgdLogisticRegressionOva => Context.MulticlassClassification.Trainers.OneVersusAll(Context.BinaryClassification.Trainers.SymbolicSgdLogisticRegression()),
            _ => throw new ArgumentException("")
        };

        var model = estimator.Fit(preprocessedTrainData);
        pfi = PermutationFeatureImportanceExtensions.PermutationFeatureImportance(Context.MulticlassClassification, model, preprocessedTrainData, permutationCount: 3);

        uint noCnt;

        WriteLineColor(" STEP 2: PFI (permutation feature importance)");
        WriteLineColor("----------------------------------------------------------------------------------");
        WriteLineColor($" PFI (by MicroAccuracy), threshold: {threshold}");
        WriteLineColor("----------------------------------------------------------------------------------");
        WriteLineColor($"  {"No",4} {"Feature",-15} {"MicroAccuracy",15} {"95% Mean",15}");
        noCnt = 1;
        var microAccuracy = pfi.Select(x => x.MicroAccuracy).ToList();
        var microSortedIndices = pfi
            .Select((metrics, index) => new { index, metrics.MicroAccuracy })
            .OrderByDescending(feature => Math.Abs(feature.MicroAccuracy.Mean))
            .Select(feature => feature.index);
        foreach (int i in microSortedIndices)
        {
            if (Math.Abs(microAccuracy[i].Mean) < threshold)
            {
                WriteLineColor($"  {noCnt++,3}. {header[i],-15} {microAccuracy[i].Mean,15:F3} {1.95 * microAccuracy[i].StandardError,15:F3} (candidate for deletion!)", ConsoleColor.Red);
            }
            else
            {
                WriteLineColor($"  {noCnt++,3}. {header[i],-15} {microAccuracy[i].Mean,15:F3} {1.95 * microAccuracy[i].StandardError,15:F3}");
            }
        }
        WriteLineColor("----------------------------------------------------------------------------------");

        var macroAccuracy = pfi.Select(x => x.MacroAccuracy).ToList();
        var macroSortedIndices = pfi
            .Select((metrics, index) => new { index, metrics.MacroAccuracy })
            .OrderByDescending(feature => Math.Abs(feature.MacroAccuracy.Mean))
            .Select(feature => feature.index);
        WriteLineColor($" PFI (by MacroAccuracy), threshold: {threshold}");
        WriteLineColor("----------------------------------------------------------------------------------");
        WriteLineColor($"  {"No",4} {"Feature",-15} {"MacroAccuracy",15} {"95% Mean",15}");
        noCnt = 1;
        foreach (int i in macroSortedIndices)
        {
            if (Math.Abs(macroAccuracy[i].Mean) < threshold)
            {
                WriteLineColor($"  {noCnt++,3}. {header[i],-15} {macroAccuracy[i].Mean,15:F3} {1.95 * macroAccuracy[i].StandardError,15:F3} (candidate for deletion!)", ConsoleColor.Red);
            }
            else
            {
                WriteLineColor($"  {noCnt++,3}. {header[i],-15} {macroAccuracy[i].Mean,15:F3} {1.95 * macroAccuracy[i].StandardError,15:F3}");
            }
        }
        WriteLineColor("----------------------------------------------------------------------------------");
    }

    internal static void CorrelationMatrix(float threshold, IDataView trainingDataView)
    {
        var trainingDataCollection = Context.Data.CreateEnumerable<ModelInput>(trainingDataView, reuseRowObject: true);
        (var header, var dataArray) = trainingDataCollection.ExtractDataAndHeader();

        var matrix = Correlation.PearsonMatrix(dataArray.ToArray());
        matrix.ToConsole(header, threshold);

        WriteLineColor("  We can remove one of the next high correlated features!");
        WriteLineColor("    - closer to  0 => low correlated features");
        WriteLineColor("    - closer to  1 => direct high correlated features");
        WriteLineColor("    - closer to -1 => inverted high correlated features");
        WriteLineColor("----------------------------------------------------------------------------------");
        WriteLineColor($"  {"No",4} {"Feature",-15} vs. {"Feature",-15} {"Rate",15}");
        uint noCnt = 1;
        for (int i = 0; i < matrix.ColumnCount; i++)
        {
            for (int j = i; j < matrix.ColumnCount; j++)
            {
                if (i != j && Math.Abs(matrix[i, j]) > threshold)
                {
                    WriteLineColor($"  {noCnt++,3}. {header[i],-15} vs. {header[j],-15} {matrix[i, j],15:F4}");
                }
            }
        }
        WriteLineColor("----------------------------------------------------------------------------------");
    }

    private static (string[] header, double[][] dataArray) ExtractDataAndHeader(this IEnumerable<ModelInput> trainingDataCollection)
    {
        var record = new ModelInput();
        var props = record.GetType().GetProperties();

        var data = new List<List<double>>();
        uint k = 0;
        foreach (var prop in props)
        {
            if (props[k].PropertyType.Name.Equals(nameof(Single)))
            {
                var arr = trainingDataCollection.Select(r => (double)(props[k].GetValue(r) as float?).Value).ToList();
                data.Add(arr);
            }
            k++;
        }
        var header = props.Where(s => s.PropertyType.Name.Equals(nameof(Single))).Select(p => p.Name).ToArray();
        var dataArray = new double[data.Count][];
        for (int i = 0; i < data.Count; i++)
        {
            dataArray[i] = data[i].ToArray();
        }

        return (header, dataArray);
    }

    internal static void Evaluate(IDataView testingData, ExperimentResult<MulticlassClassificationMetrics> experimentResult)
    {
        var predictions = experimentResult.BestRun.Model.Transform(testingData);
        var metrics = Context.MulticlassClassification.Evaluate(predictions, Label, Score, PredictedLabel);
        PrintMultiClassClassificationMetrics(metrics, true);
    }

    internal static void CrossValidate(IDataView data, ExperimentResult<MulticlassClassificationMetrics> experimentResult)
    {
        var crossValidationResults = Context.MulticlassClassification.CrossValidate(data, experimentResult.BestRun.Estimator, 5);
        PrintMulticlassClassificationFoldsAverageMetrics(crossValidationResults);
    }
}
