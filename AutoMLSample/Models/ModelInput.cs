namespace AutoMLSample.Models;

internal class ModelInput
{
    [LoadColumn(0)]
    public float Temperature { get; set; }

    [LoadColumn(1)]
    public float Temperature2 { get; set; }

    [LoadColumn(2)]
    public float Luminosity { get; set; }

    [LoadColumn(3)]
    public float Infrared { get; set; }

    [LoadColumn(4)]
    public float Distance { get; set; }

    [LoadColumn(5)]
    public float PIR { get; set; }

    [LoadColumn(6)]
    public float Humidity { get; set; }

    [ColumnName("Label"), LoadColumn(8)]
    public string Label { get; set; }
}
