using UnityEngine;

public class MaintainWorldScale : MonoBehaviour
{
    private Vector3 initialParentScale;

    void Start()
    {
        // Store the initial parent's scale
        if (transform.parent != null)
        {
            initialParentScale = transform.parent.localScale;
        }
        else
        {
            Debug.LogError("This object does not have a parent.");
        }
    }

    void Update()
    {
        // Check if the parent's scale has changed
        if (transform.parent != null && transform.parent.localScale != initialParentScale)
        {
            AdjustScale();
            initialParentScale = transform.parent.localScale;
        }
    }

    void AdjustScale()
    {
        // Calculate the required local scale to maintain a world scale of 1
        Vector3 parentScale = transform.parent.localScale;
        Vector3 newLocalScale = new Vector3(
            1.0f / parentScale.x,
            1.0f / parentScale.y,
            1.0f / parentScale.z
        );

        transform.localScale = newLocalScale;
    }
}
