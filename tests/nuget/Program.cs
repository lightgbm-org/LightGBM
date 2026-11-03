using System.Runtime.InteropServices;

// Nonconstant labels ensure an untrained model cannot pass.
var features = Enumerable.Range(0, 64).Select(i => (float)(i % 4)).ToArray();
var labels = features.Select(x => 2 * x + 1).ToArray();
const string parameters = "objective=regression num_leaves=4 min_data_in_leaf=1 min_data_in_bin=1 learning_rate=0.2 num_threads=2 verbosity=-1";
IntPtr dataset = IntPtr.Zero;
IntPtr booster = IntPtr.Zero;
try
{
    Check(Native.LGBM_DatasetCreateFromMat(features, 0, features.Length, 1, 1, parameters, IntPtr.Zero, out dataset));
    Check(Native.LGBM_DatasetSetField(dataset, "label", labels, labels.Length, 0));
    Check(Native.LGBM_BoosterCreate(dataset, parameters, out booster));
    for (int i = 0; i < 30; ++i)
    {
        Check(Native.LGBM_BoosterUpdateOneIter(booster, out _));
    }

    float[] input = { 0, 1, 2, 3 };
    double[] predictions = new double[input.Length];
    Check(Native.LGBM_BoosterPredictForMat(
        booster, input, 0, input.Length, 1, 1, 0, 0, -1, "num_threads=2", out long length, predictions));
    if (length != predictions.Length)
    {
        throw new InvalidOperationException($"Expected {predictions.Length} predictions, got {length}.");
    }
    for (int i = 0; i < predictions.Length; ++i)
    {
        double expected = 2 * input[i] + 1;
        // Allow small floating-point differences instead of requiring exact predictions.
        if (!double.IsFinite(predictions[i]) || Math.Abs(predictions[i] - expected) > 0.05)
        {
            throw new InvalidOperationException($"Expected prediction near {expected}, got {predictions[i]}.");
        }
    }
}
finally
{
    // Free the model and its dataset.
    try
    {
        if (booster != IntPtr.Zero)
        {
            Check(Native.LGBM_BoosterFree(booster));
        }
    }
    finally
    {
        if (dataset != IntPtr.Zero)
        {
            Check(Native.LGBM_DatasetFree(dataset));
        }
    }
}
Console.WriteLine($"NuGet regression smoke test passed ({RuntimeInformation.ProcessArchitecture}).");

static void Check(int result)
{
    if (result != 0)
    {
        throw new InvalidOperationException(Marshal.PtrToStringUTF8(Native.LGBM_GetLastError()));
    }
}

// Test the native library supplied by NuGet.
internal static class Native
{
    private const string Library = "lib_lightgbm";

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr LGBM_GetLastError();

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int LGBM_DatasetCreateFromMat(
        [In] float[] data, int dataType, int rows, int columns, int rowMajor,
        string parameters, IntPtr reference, out IntPtr dataset);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int LGBM_DatasetSetField(IntPtr dataset, string field, [In] float[] data, int count, int dataType);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int LGBM_BoosterCreate(IntPtr dataset, string parameters, out IntPtr booster);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int LGBM_BoosterUpdateOneIter(IntPtr booster, out int producedEmptyTree);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int LGBM_BoosterPredictForMat(
        IntPtr booster, [In] float[] data, int dataType, int rows, int columns, int rowMajor,
        int predictType, int startIteration, int numIterations, string parameters, out long length, [Out] double[] result);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int LGBM_BoosterFree(IntPtr booster);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int LGBM_DatasetFree(IntPtr dataset);
}
