using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;

/* Class representing numerical and visual points in a cloud for
 * a GameObject. Attach to any relevant GameObject where a transform
 * is planned to be registered.
 */
public class PointCloud  : MonoBehaviour
{
    [SerializeField]
    // Sphere generator
    GameObject spherePrefab;
    
    // Internal list of points in the cloud
    List<GameObject> spheres;

    // Start is called before the first frame update
    void Start()
    {
        spheres = new List<GameObject>();
    }

    // Instantiate and add a new point to the cloud
    public void AddSphere(Vector3 pos)
    {
        GameObject marker = Instantiate(spherePrefab, pos, Quaternion.identity);
        marker.transform.parent = gameObject.transform;
        spheres.Add(marker);
        
    }

    // Delete the closest sphere from the given position
    public void DeleteClosestSphere(Vector3 pos)
    {
        float smallestDistance = 1000000;
        GameObject smallestSphere = null;

        // Linear search of the closest sphere in the cloud
        foreach (GameObject s in spheres)
        {
            float dist = Vector3.Distance(s.transform.position, pos);
            if (dist < smallestDistance)
            {
                smallestSphere = s;
                smallestDistance = dist;
            }
        }
        
        // Delete the sphere if it is close enough
        if (smallestSphere != null && smallestDistance < 0.0175f)
        {
            spheres.Remove(smallestSphere);
            Destroy(smallestSphere);
        }
    }

    // Put all children Transforms in a Math.Net Vector list
    public List<Vector<double>> TransVecList()
    {
        List<Vector<double>> vecList = new List<Vector<double>>();
        foreach (Transform fidsTrans in this.GetComponentInChildren<Transform>())
        {
            Vector<double> tmpVec = Vector<double>.Build.Dense(3);
            tmpVec[0] = fidsTrans.position.x;
            tmpVec[1] = fidsTrans.position.y;
            tmpVec[2] = fidsTrans.position.z;
            vecList.Add(tmpVec);
        }
        return vecList;
    }

    public List<Vector3> TransVecListUnity()
    {
        List<Vector3> vecList = new List<Vector3>();
        foreach (Transform childTransform in transform)
        {
            vecList.Add(childTransform.position);
        }
        return vecList;
    }

}
