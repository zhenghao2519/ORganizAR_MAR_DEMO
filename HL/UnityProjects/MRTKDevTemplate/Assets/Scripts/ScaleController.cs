using UnityEngine;

public class ScaleController : MonoBehaviour
{
    private Vector3 originalScale;
    public float percent;
    void Start()
    {

        originalScale = transform.localScale;
    }

    public void ScaleByPercent()
    {
        transform.localScale = originalScale * (1 + percent / 100.0f);
    }

    public void ResetScale()
    {
        transform.localScale = originalScale;
    }
}
