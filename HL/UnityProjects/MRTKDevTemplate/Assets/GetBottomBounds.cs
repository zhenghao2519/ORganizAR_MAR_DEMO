using UnityEngine;

public class GetBottomBounds : MonoBehaviour
{

    public void GetBottomCorners()
    {
        MeshFilter meshFilter = GetComponent<MeshFilter>();
        if (meshFilter == null)
        {
            Debug.LogError("MeshFilter not found!");
            return;
        }

        // Get the mesh bounds in local space
        Bounds bounds = meshFilter.mesh.bounds;

        // Get the 4 bottom corners of the bounding box
        Vector3[] corners = new Vector3[4];

        // Calculate the corners in local space
        Vector3 center = bounds.center;
        Vector3 extents = bounds.extents;

        corners[0] = center + new Vector3(-extents.x, -extents.y, -extents.z);
        corners[1] = center + new Vector3(extents.x, -extents.y, -extents.z);
       
        corners[4] = center + new Vector3(-extents.x, extents.y, -extents.z);
        corners[5] = center + new Vector3(extents.x, extents.y, -extents.z);
    

        // Transform corners to world space
        for (int i = 0; i < corners.Length; i++)
        {
            corners[i] = transform.TransformPoint(corners[i]);
        }

    }

  
}
