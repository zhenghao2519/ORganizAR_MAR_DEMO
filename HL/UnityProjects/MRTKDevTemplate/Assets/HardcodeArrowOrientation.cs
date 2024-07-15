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
            // arrow tip at median //TODO python side use top of bbox
            transform.Translate(0.1255f, 0, 0, Space.Self);
            

        }
    }
    private void AlignYAxisWithWorld()
    {
        transform.rotation = Quaternion.FromToRotation(transform.right, Vector3.up) * transform.rotation;
    }
}
