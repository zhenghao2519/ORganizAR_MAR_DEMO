using UnityEngine;

public class TintController : MonoBehaviour
{
    public float tintValue; 
    private Renderer modelRenderer;

    public void Start()
    {
        this.modelRenderer = GetComponent<Renderer>();
    }
    void Update()
    {
        Color newColor = Color.Lerp(Color.red, Color.green, tintValue);
        if (modelRenderer != null)
        {
            modelRenderer.material.color = newColor;
        }
    }
}
