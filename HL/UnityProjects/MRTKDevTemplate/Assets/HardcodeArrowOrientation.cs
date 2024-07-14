using UnityEngine;

public class HardcodeArrowOrientation : MonoBehaviour
{
    private Vector3 lastPosition;
    private bool first = true;

    // Start is called before the first frame update
    void Start()
    {
        lastPosition = transform.position;
    }

    // Update is called once per frame
    void Update()
    {
        if (transform.position != Vector3.zero && first)
        {
            first = false;
            AlignYAxisWithWorld();
            lastPosition = transform.position;
        }
    }

    private void AlignYAxisWithWorld()
    {
        transform.rotation = Quaternion.FromToRotation(transform.right, Vector3.up) * transform.rotation;
    }
}
