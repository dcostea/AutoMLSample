namespace AutoMLSample.Models;

internal class ModelOutput
{
    [ColumnName("PredictedLabel")]
    public string PredictedLabel;

    [ColumnName("Score")]
    public float[] Score;
}
