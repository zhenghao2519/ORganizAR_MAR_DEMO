using System;
using UnityEngine;

public class CopyTransform : MonoBehaviour
{
    public GameObject source;
    private bool stopCopying = false;

    public void StopCopying() {
        stopCopying = true;
    }


    void Update()
    {
        if (source != null && !stopCopying)
        {
            transform.position = source.transform.position;
        }
    }
}
