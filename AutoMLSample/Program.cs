using AutoMLSample.Services;

namespace AutoMLSample;

internal static class Program
{
    private static async Task Main(string[] args)
    {
        Log.Logger = new LoggerConfiguration()
            .MinimumLevel.Debug()
            .MinimumLevel.Override("Microsoft", Serilog.Events.LogEventLevel.Warning)
            .MinimumLevel.Override("System", Serilog.Events.LogEventLevel.Warning)
            .WriteTo.Console(outputTemplate: "[{Timestamp:HH:mm:ss} {Level:u3}] {Message:lj}{NewLine}")
            .WriteTo.File("logs/check.log", Serilog.Events.LogEventLevel.Debug, "[{Timestamp:HH:mm:ss} {Level:u3}] {Message:lj}{NewLine}{Exception}")
            .CreateBootstrapLogger();

        var data = MachineLearningServices.LoadData();
        (var trainingData, var testingData) = data.SplitData();

        var experimentResult1 = await MachineLearningServices.AutoTrainAsync(30, trainingData, testingData);
        Console.WriteLine($" Trial metric: {experimentResult1.Metric}");
        Console.WriteLine("----------------------------------------------------------------------------------");

        var experimentResult = MachineLearningServices.AutoTrain(30, trainingData);
        MachineLearningServices.Evaluate(testingData, experimentResult);
        MachineLearningServices.CrossValidate(data, experimentResult);
        MachineLearningServices.PFI(0.01F, experimentResult.BestRun.TrainerName, trainingData);
        MachineLearningServices.CorrelationMatrix(0.9F, trainingData);
    }
}
