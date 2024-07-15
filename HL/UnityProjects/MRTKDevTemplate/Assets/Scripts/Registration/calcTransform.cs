using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics;
using static UnityEditor.Experimental.AssetDatabaseExperimental.AssetDatabaseCounters;

public class calcTransform : MonoBehaviour
{
    //private Matrix<double> covMatrix = Matrix<double>.Build.Dense(3, 3);
    //private List<Vector<double>> vecListSrc = new List<Vector<double>>();
    //private List<Vector<double>> vecListSrcCoM = new List<Vector<double>>();
    //private List<Vector<double>> vecListDst = new List<Vector<double>>();
    //private List<Vector<double>> vecListDstCoM = new List<Vector<double>>();

    [SerializeField]
    GameObject srcContainer, dstContainer;
    public bool register = false;
    public float maxDistance = 0.01f;
    private int counter = 0;

    public void Register() {
        register = true;

    }

    private void LateUpdate()
    {
        if (register) {
            register = false;
            Debug.Log("running register");

            ComputeAndApplyTransform();
            counter++;
            if (counter == 10) {
                register = false;
                counter = 0;
                return;
            }
            List<Vector3> vecListSrc = srcContainer.GetComponent<PointCloud>().TransVecListUnity();
            List<Vector3> vecListDst = dstContainer.GetComponent<PointCloud>().TransVecListUnity();
            
            for (int i = 0; i < vecListSrc.Count; i++)
            {
                if (Vector3.Distance(vecListSrc[i], vecListDst[i]) > maxDistance)
                {
                    Debug.Log(Vector3.Distance(vecListSrc[i], vecListDst[i]) > maxDistance);
                    register = true;
                    return;
                }
            }
          



        }
        
    }
    //public GameObject testFidSrc, testFidDst;

    private void Start()
    {

        //for(int i= 0; i<4; i++)
        //    ComputeAndApplyTransformICP();
    }

    // Use this for initialization
    public void ComputeAndApplyTransform()
    {
        List<Vector<double>> vecListSrc = srcContainer.GetComponent<PointCloud>().TransVecList();
        List<Vector<double>> vecListDst = dstContainer.GetComponent<PointCloud>().TransVecList();

        // calculate center of mass of Vector list
        Vector<double> coMSrc = CalcCoM(vecListSrc);
        Vector<double> coMDst = CalcCoM(vecListDst);

        // for debugging only
        //CreateMarker(new Vector3((float)coMSrc[0], (float) coMSrc[1], (float) coMSrc[2]), srcContainer);
        //CreateMarker(new Vector3((float) coMDst[0], (float) coMDst[1], (float) coMDst[2]), dstContainer);

        // calculate vectors translated to center of mass
        List<Vector<double>> vecListSrcCoM = CalcCoMVecs(vecListSrc, coMSrc);
        List<Vector<double>> vecListDstCoM = CalcCoMVecs(vecListDst, coMDst);

        // for debugging only
        //foreach (Vector<double> vec in vecListSrcCoM)
        //{
        //    CreateMarker(new Vector3((float)vec[0], (float)vec[1], (float)vec[2]), srcContainer);
        //}
        //foreach (Vector<double> vec in vecListDstCoM)
        //{
        //    CreateMarker(new Vector3((float)vec[0], (float)vec[1], (float)vec[2]), dstContainer);
        //}

        // calculate covariance matrix
        Matrix<double> covMatrix = CalcCov(vecListDstCoM, vecListSrcCoM);

        // perform singular value decomposition
        var svd = covMatrix.Svd(true);

        // get the rotation through R = U VT
        var rot = svd.U * svd.VT;
        var tra = coMDst - rot * coMSrc;

        // convert Math rotation and translation to Matrix4x4
        Quaternion rotMat = QuaternionFromMatrix(MathRot2Mat4x4(rot, tra));


        // apply rotation and transformation to src gameobject
        //RotandTrans(testFidSrc, testFidDst, rot, tra);

        RotandTransRigid(srcContainer, dstContainer, rotMat, rot, tra);
        Debug.Log("end");
    }
    public void ComputeAndApplyTransformICP()
    {
        List<Vector<double>> vecListSrc = srcContainer.GetComponent<PointCloud>().TransVecList();
        List<Vector<double>> vecListDst = dstContainer.GetComponent<PointCloud>().TransVecList();

        // calculate center of mass of Vector list
        Vector<double> coMSrc = CalcCoM(vecListSrc);
        Vector<double> coMDst = CalcCoM(vecListDst);

        // calculate vectors translated to center of mass
        List<Vector<double>> vecListSrcCoM = CalcCoMVecs(vecListSrc, coMSrc);
        List<Vector<double>> vecListDstCoM = CalcCoMVecs(vecListDst, coMDst);

        List<int> pairedIndices = MatchPoints(vecListSrcCoM, vecListDstCoM);

        List<Vector<double>> vecListDstCoMPaired = new List<Vector<double>>();
        for (int j = 0; j < pairedIndices.Count; j++)
        {
            Debug.Log(pairedIndices[j]);
            vecListDstCoMPaired.Add(vecListDstCoM[pairedIndices[j]]);
        }

        Debug.Log(vecListDstCoMPaired.Count);
        // calculate covariance matrix
        Matrix<double> covMatrix = CalcCov(vecListDstCoMPaired,vecListSrcCoM);

        // perform singular value decomposition
        var svd = covMatrix.Svd(true);

        // get the rotation through R = U VT
        var rot = svd.U * svd.VT;
        var tra = coMDst - rot * coMSrc;

        // convert Math rotation and translation to Matrix4x4
        Quaternion rotMat = QuaternionFromMatrix(MathRot2Mat4x4(rot, tra));


        // apply rotation and transformation to src gameobject
        //RotandTrans(testFidSrc, testFidDst, rot, tra);

        RotandTransRigid(srcContainer, dstContainer, rotMat, rot, tra);

    }

    public List<int> MatchPoints(List<Vector<double>> list1, List<Vector<double>> list2)
    {
        List<int> indexPool = new List<int>();
        List<int> pairs = new List<int>();
        for (int i = 0; i < list2.Count; i++)
            indexPool.Add(i);

        // Iterate over each point
        for (int i = 0; i < list1.Count; i++)
        {
            int closestIdx = indexPool[0];
            double minDist = Distance.Euclidean(list1[i], list2[closestIdx]);

            // Iterate over the index pool
            foreach (int j in indexPool)
            {
               
                double distIJ = Distance.Euclidean(list1[i], list2[j]);
                closestIdx = (distIJ < minDist) ? j : closestIdx;
                minDist = (distIJ < minDist) ? distIJ : minDist;
            }
            // Add index pair, remove closest point index from pool
            pairs.Add(closestIdx);
            indexPool.Remove(closestIdx);
        }
        return pairs;
    }

    // calculate vectors translated to center of mass
    List<Vector<double>> CalcCoMVecs(List<Vector<double>> fids, Vector<double> coM)
    {
        List<Vector<double>> vecListCoM = new List<Vector<double>>();
        foreach (Vector<double> vec in fids)
        {
            vecListCoM.Add(vec - coM);
        }
        return vecListCoM;
    }

    // calculate covariance matrix
    Matrix<double> CalcCov (List<Vector<double>> srcVecs, List<Vector<double>> dstVecs)
    {
        Matrix<double> covMatrix = Matrix<double>.Build.Dense(3, 3);
        for (int i = 0; i < srcVecs.Count; i++)

        {
            covMatrix += ColVec2Mat(srcVecs[i]) * RowVec2Mat(dstVecs[i]);
        }
        return covMatrix;
    }

    void RotandTransRigid(GameObject srcObj, GameObject dstObj, Quaternion rotMat4, Matrix<double> rot, Vector<double> tra)
    {
        Vector<double> srcObjPos = Vector<double>.Build.Dense(3);
        srcObjPos[0] = srcObj.transform.position.x;
        srcObjPos[1] = srcObj.transform.position.y;
        srcObjPos[2] = srcObj.transform.position.z;
        Vector<double> dstObjPos = Vector<double>.Build.Dense(3);
        dstObjPos[0] = dstObj.transform.position.x;
        dstObjPos[1] = dstObj.transform.position.y;
        dstObjPos[2] = dstObj.transform.position.z;

        // All 
        Vector<double> transVec = rot * srcObjPos + tra;
        //Matrix<double> transVecM = rot * ColVec2Mat(dstObjPos) + ColVec2Mat(tra);
        //Matrix<double> transVecMT = rot * RowVec2Mat(dstObjPos) + ColVec2Mat(tra);
        //Matrix<double> transVecMb = rot * ColVec2Mat(dstObjPos) + RowVec2Mat(tra);
        //Matrix<double> transVecMbT = rot * RowVec2Mat(dstObjPos) + RowVec2Mat(tra);
        Vector<double> newObjPos = transVec;
        srcObj.transform.position = new Vector3((float)newObjPos[0], (float)newObjPos[1], (float)newObjPos[2]);
        srcObj.transform.rotation = rotMat4 * srcObj.transform.rotation;
    }

    void RotandTrans (GameObject srcObj, GameObject dstObj, Matrix<double> rot, Vector<double> tra)
    {
        Vector<double> srcObjPos = Vector<double>.Build.Dense(3);
        srcObjPos[0] = srcObj.transform.position.x;
        srcObjPos[1] = srcObj.transform.position.y;
        srcObjPos[2] = srcObj.transform.position.z;
        Vector<double> dstObjPos = Vector<double>.Build.Dense(3);
        dstObjPos[0] = dstObj.transform.position.x;
        dstObjPos[1] = dstObj.transform.position.y;
        dstObjPos[2] = dstObj.transform.position.z;

        // All 
        Vector<double> transVec = rot * srcObjPos + tra;
        Vector<double> newObjPos = transVec; 
        dstObj.transform.position = new Vector3((float)newObjPos[0], (float) newObjPos[1], (float) newObjPos[2]);
    }

    // create spherical marker
    void CreateMarker(Vector3 pos, GameObject parentObj)
    {
        var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere); // make sphere
        sphere.transform.parent = parentObj.transform; // !!! set parent before changin position, otherwise collider error! Unity 2018.1.9 BUG
        sphere.transform.localScale = Vector3.one * 0.1f; // scale sphere down
        sphere.transform.position = pos; // position sphere at cursor position
    }

    //// static functions
    ///     // convert column vector to matrix that has the vector in col 1 and 0 everywhere else
    static Matrix<double> ColVec2Mat(Vector<double> vec)
    {
        Matrix<double> colVecMatrix = Matrix<double>.Build.Dense(3, 3);
        colVecMatrix[0, 0] = vec[0];
        colVecMatrix[1, 0] = vec[1];
        colVecMatrix[2, 0] = vec[2];
        return colVecMatrix;
    }

    // convert row vector to matrix that has the vector in row 1 and 0 everywhere else
    static Matrix<double> RowVec2Mat(Vector<double> vec)
    {
        Matrix<double> rowVecMatrix = Matrix<double>.Build.Dense(3, 3);
        rowVecMatrix[0, 0] = vec[0];
        rowVecMatrix[0, 1] = vec[1];
        rowVecMatrix[0, 2] = vec[2];
        return rowVecMatrix;
    }

    static Quaternion QuaternionFromMatrix(Matrix4x4 m)
    {
        // Adapted from: http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
        Quaternion q = new Quaternion();
        q.w = Mathf.Sqrt(Mathf.Max(0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2;
        q.x = Mathf.Sqrt(Mathf.Max(0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2;
        q.y = Mathf.Sqrt(Mathf.Max(0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2;
        q.z = Mathf.Sqrt(Mathf.Max(0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2;
        q.x *= Mathf.Sign(q.x * (m[2, 1] - m[1, 2]));
        q.y *= Mathf.Sign(q.y * (m[0, 2] - m[2, 0]));
        q.z *= Mathf.Sign(q.z * (m[1, 0] - m[0, 1]));
        return q;
    }

    //public static Quaternion QuaternionFromMatrix(Matrix4x4 m) 
    //{ 
    //    return Quaternion.LookRotation(m.GetColumn(2), m.GetColumn(1)); 
    //}

    // calculate center of mass of Vector list
    static Vector<double> CalcCoM(List<Vector<double>> Fids)
    {
        Vector<double> coM = Vector<double>.Build.Dense(3);
        foreach (Vector<double> vec in Fids)
            coM += vec;
        coM = coM.Divide(Fids.Count);
        return coM;
    }

    static Matrix4x4 MathRot2Mat4x4(Matrix<double> rotmat, Vector<double> travec)
    {
        Matrix4x4 mat4 = new Matrix4x4();
        mat4.m00 = (float)rotmat[0, 0];
        mat4.m01 = (float)rotmat[0, 1];
        mat4.m02 = (float)rotmat[0, 2];
        mat4.m03 = (float)travec[0];
        mat4.m10 = (float)rotmat[1, 0];
        mat4.m11 = (float)rotmat[1, 1];
        mat4.m12 = (float)rotmat[1, 2];
        mat4.m13 = (float)travec[1];
        mat4.m20 = (float)rotmat[2, 0];
        mat4.m21 = (float)rotmat[2, 1];
        mat4.m22 = (float)rotmat[2, 2];
        mat4.m23 = (float)travec[2];
        mat4.m30 = 0;
        mat4.m31 = 0;
        mat4.m32 = 0;
        mat4.m33 = 1;
        //Debug.Log("mat4 " + mat4.ToString());
        //Debug.Log("rotmat " + rotmat.ToMatrixString());
        return mat4;

    }

}
