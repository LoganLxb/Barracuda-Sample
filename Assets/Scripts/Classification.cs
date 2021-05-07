using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System.Linq;
using UnityEngine.UI;
public class Classification : MonoBehaviour
{
    public NNModel NNModel;
    public TextAsset Lable;
    public UnityEngine.UI.Text Result;
    public Preprocess preprocess;
    private const int ImageSize = 224;
    private const string INPUT_NAME = "images";
    private const string OUTPUT_NAME = "Softmax";
    private Model Runtimemodel;
    private IWorker Worker;
    private string[] labels;
    // Start is called before the first frame update

    private void Awake()
    {
        Runtimemodel = ModelLoader.Load(NNModel);
        Worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, Runtimemodel);
        Loadable();
    }
    void Start()
    {
        RawImage image= GetComponent<RawImage>();
        //Texture2D texture = image.mainTexture as Texture2D;
        preprocess.ScaleAndCropImage(image, ImageSize, RunModel);
    }


    void Loadable()
    {
        //get only items in quotes
        var stringArray = Lable.text.Split('"').Where((item, index) => index % 2 != 0);
        Debug.Log(stringArray);
        //get every other item
        labels = stringArray.Where((x, i) => i % 2 != 0).ToArray();
        
    }
    // Update is called once per frame
    void Update()
    {
        
    }

    void RunModel(byte[] pixels)
    {
        StartCoroutine(RunModelRoutine(pixels));
    }

    IEnumerator RunModelRoutine(byte[] pixels)
    {
        Tensor tensor = TransformInput(pixels);
        var input = new Dictionary<string, Tensor>()
        {
            { INPUT_NAME, tensor }
        };
        Worker.Execute(input);
        Tensor output = Worker.PeekOutput(OUTPUT_NAME);

        List<float> temp = output.ToReadOnlyArray().ToList();
        float max = temp.Max();
        int index = temp.IndexOf(max);

        //set ui

        string lable = labels[index];
        Debug.Log(lable);
        Result.text = lable;

        tensor.Dispose();
        output.Dispose();
        yield return null;
        
    }

    Tensor TransformInput(byte[] pixels)
    {
        float[] transformedPixels = new float[pixels.Length];

        for (int i = 0; i < pixels.Length; i++)
        {
            transformedPixels[i] = (pixels[i] - 127f) / 128f;
        }
        return new Tensor(1, ImageSize, ImageSize, 3, transformedPixels);
    }

    private byte[] duplicateTexture(Texture2D source)
    {
        RenderTexture renderTex = RenderTexture.GetTemporary(
                    source.width,
                    source.height,
                    0,
                    RenderTextureFormat.Default,
                    RenderTextureReadWrite.Linear);

        Graphics.Blit(source, renderTex);
        RenderTexture previous = RenderTexture.active;
        RenderTexture.active = renderTex;
        Texture2D readableText = new Texture2D(source.width, source.height);
        readableText.ReadPixels(new Rect(0, 0, renderTex.width, renderTex.height), 0, 0);
        readableText.Apply();
        //这里可以转 JPG PNG EXR  Unity都封装了固定的Api
        byte[] bytes = readableText.EncodeToJPG();
        return bytes;
        //RenderTexture.active = previous;
        //RenderTexture.ReleaseTemporary(renderTex);
        //return readableText;
    }
}
