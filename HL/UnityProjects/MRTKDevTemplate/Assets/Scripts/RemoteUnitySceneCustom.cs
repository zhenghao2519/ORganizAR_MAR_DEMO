
using System;
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using MixedReality.Toolkit.Audio;
using MixedReality.Toolkit.Examples;
using UnityEngine.WSA;
using UnityEngine.UIElements;
using System.Net.NetworkInformation;
using static UnityEngine.XR.Interaction.Toolkit.Inputs.Interactions.SectorInteraction;
using static Microsoft.MixedReality.GraphicsTools.MeshInstancer;
using System.Reflection;

public class RemoteUnitySceneCustom : MonoBehaviour
{
    public GameObject m_tts;

    private Dictionary<int, GameObject> m_remote_objects;
    private bool m_loop;
    private bool m_mode;
    private int m_last_key;
    public List<GameObject> targets = new List<GameObject>();

    public int detectionCount = 10;
    private bool done = false;
    private int pc_counter = 0;
    public bool AIDone = false;
    public HandMenuManager HandMenuManager;

    public GameObject SelectedSetup;
    public GameObject table1;
    public GameObject table2;

    private List<Color> colorList = new List<Color>
    {
        Color.red,
        Color.green,
        Color.blue,
        Color.yellow,
        Color.cyan,
        Color.magenta,
        Color.white,
        Color.gray
        
    };

    [Tooltip("Set to BasicMaterial to support semi-transparent primitives.")]
    public Material m_material;

    void Start()
    {
        

        m_remote_objects = new Dictionary<int, GameObject>();
        m_loop = false;
        m_mode = false;
    }
    public GameObject targetRenderingsParent;
    void Update()
    {
        while (GetMessage() && m_loop) ;
        if (AIDone)
        {
            AIDone = false; //for debugging
            targetRenderingsParent.SetActive(true);
            HandMenuManager.AIDone();
        }
    }
    public void SetTargetsActive() {

        targetRenderingsParent.SetActive(true);
    }
  
    bool GetMessage()
    {
        uint command;
        byte[] data;
        if (!hl2ss.PullMessage(out command, out data)) { return false; }
        hl2ss.PushResult(ProcessMessage(command, data));
        hl2ss.AcknowledgeMessage(command);
        return true;
    }

    uint ProcessMessage(uint command, byte[] data)
    {
        uint ret = 0;

        switch (command)
        {
            case 0: ret = MSG_CreatePrimitive(data); break;
            case 1: ret = MSG_SetActive(data); break;
            case 2: ret = MSG_SetWorldTransform(data); break;
            case 3: ret = MSG_SetLocalTransform(data); break;
            case 4: ret = MSG_SetColor(data); break;
            case 5: ret = MSG_SetTexture(data); break;
            case 6: ret = MSG_CreateText(data); break;
            case 7: ret = MSG_SetText(data); break;
            case 8: ret = MSG_Say(data); break;

            case 16: ret = MSG_Remove(data); break;
            case 17: ret = MSG_RemoveAll(data); break;
            case 18: ret = MSG_BeginDisplayList(data); break;
            case 19: ret = MSG_EndDisplayList(data); break;
            case 20: ret = MSG_SetTargetMode(data); break;
            case 21: ret = MSG_SpawnArrow(data); break;
            case 22: ret = MSG_ReceiveDetection(data); break;
            case 23: ret = MSG_CheckDone(); break;
            case 24: ret = MSG_CreatePCRenderer(data); break;
            case 25: ret = MSG_SetPointCloud(data); break;
            case 26: ret = GetTargetPosition(data); break;
            case 27: ret = ApplyTableScale(data); break;
            case ~0U: ret = MSG_Disconnect(data); break;
        }

        return ret;
    }

    public uint ApplyTableScale(byte[] data)
    {
        if (data.Length < 4)
        {
            return 0;
        }

        int ratio = BitConverter.ToInt32(data, 0);
        double scalingRatio = (double)ratio / 100;
        ScaleObject(table1, scalingRatio);
        ScaleObject(table2, scalingRatio);

        return 1;
    }

    private void ScaleObject(GameObject obj, double scalingRatio)
    {
        if (obj != null)
        {
            obj.transform.localScale = new Vector3(
                (float)(obj.transform.localScale.x * scalingRatio),
                (float)(obj.transform.localScale.y * scalingRatio),
                (float)(obj.transform.localScale.z * scalingRatio)
            );
        }
    }

    static uint FloatToUint(float value)
    {

        return (uint)((value + 1000.0f) * 1000);
    }

    uint GetTargetPosition(byte[] data)
    {
        if (data.Length < 8) { return 0; }

        int promptIndex = BitConverter.ToInt32(data, 0);
        int axis = BitConverter.ToInt32(data, 4);
        List<GameObject> childObjects = new List<GameObject>();
        foreach (Transform child in SelectedSetup.GetComponentsInChildren<Transform>())
        {
            if (child.gameObject != SelectedSetup)
            {
                childObjects.Add(child.gameObject);
            }
        }
        GameObject go = childObjects[promptIndex];
        if ( go == null)
        {
            return 100000000;
        }
        Vector3 targetPos  = go.transform.position;
        uint result = 0;
        switch (axis) 
        {
            case 0:
                result = FloatToUint(targetPos.x);
                break;
            case 1:
                result = FloatToUint(targetPos.y);
                break;
            case 2:
                result = FloatToUint(targetPos.z);
                break;
        }
 
        return result;
    }
    private int detections = 0;

    uint MSG_CreatePCRenderer(byte[] data)
    {
       
        if (data.Length < 8)
        {
            return 0;
        }
        detections = BitConverter.ToInt32(data, 0);

        

        pc_counter += 1;
        if (pc_counter == 4) {
            return 0;
            
        }
        GameObject pcPrefab = Resources.Load<GameObject>("PointCloudRenderer");
        GameObject pcInstance = GameObject.Instantiate(pcPrefab, Vector3.zero, Quaternion.identity);


        Renderer pcRenderer = pcInstance.GetComponent<Renderer>();
        if (pcRenderer != null)
        {
            pcRenderer.enabled = true;
        }

        // Parse index from the byte array
        int index = BitConverter.ToInt32(data, 4);

        // Get the children of SelectedSetup
        Transform cArm = SelectedSetup.transform.GetChild(0); // C-Arm
        Transform laparoscopicTower4 = SelectedSetup.transform.GetChild(1); // LaparoscopicTower4
        Transform usTower = SelectedSetup.transform.GetChild(2); // USTower

        // Set the parent based on the index
        Transform parentTransform = null;
        switch (index)
        {
            case 0:
                parentTransform = cArm;
                parentTransform.gameObject.SetActive(true);
                break;
            case 1:
                parentTransform = laparoscopicTower4;
                parentTransform.gameObject.SetActive(false);
                break;
            case 2:
                parentTransform = usTower;
                parentTransform.gameObject.SetActive(false);
                break;
            default:
                Debug.LogError("Invalid index value!");
                return 0;
        }

        // Set the parent and ensure the transform is identity in the world
        pcInstance.transform.SetParent(parentTransform, false);
        pcInstance.transform.localPosition = Vector3.zero;
        pcInstance.transform.localRotation = Quaternion.identity;
        pcInstance.transform.localScale = Vector3.one;

        


        




        return AddGameObject(pcInstance);
    }


    uint MSG_SetPointCloud(byte[] data)
    {
        if (data.Length < 4) { return 0; }
        //not more than 3 objects
        if (pc_counter == 4)
        {
            return 1;

        }
        GameObject go;
        //mode last no key used
        if (!m_remote_objects.TryGetValue(GetKey(data), out go)) { return 0; }

        PointCloudRenderer pointCloudRenderer = go.GetComponent<PointCloudRenderer>();
        // first el is len
        int nPoints = BitConverter.ToInt32(data, 0);
        Vector3[] arrVertices = new Vector3[nPoints];
        

        if (data.Length > 4)
        {
            byte[] pointsInBytes = new byte[data.Length - 4];
            Array.Copy(data, 4, pointsInBytes, 0, pointsInBytes.Length); //probably not needed just use offset below


            int floatSize = 4; // Size of a float in bytes
            for (int i = 0; i < nPoints; ++i)
            {
                float x = BitConverter.ToSingle(pointsInBytes, i * 3 * floatSize);
                float y = BitConverter.ToSingle(pointsInBytes, i * 3 * floatSize + floatSize);
                float z = BitConverter.ToSingle(pointsInBytes, i * 3 * floatSize + 2 * floatSize);
                arrVertices[i] = new Vector3(x, y, z);
            }

        }
        pointCloudRenderer.Init();
        Color color = colorList[pc_counter % colorList.Count];
        pointCloudRenderer.Render(arrVertices, color);
        if (pc_counter == detections) {
            AIDone = true;
            
        }
       
        return 1;
    }

    public void SetDone() {
        done = true;
    }
    uint MSG_CheckDone()
    {
        if (done)
        {
            return 2;
        }
        else {
            return 0;
        } 
    }


    uint MSG_ReceiveDetection(byte[] data)
    {
        if (data.Length < 4) { return 0; }
        float inc = 1.0f / detectionCount;
        int index = (BitConverter.ToInt32(data, 0));
        if (index < targets.Count)
        {
            targets[index].GetComponent<TintController>().tintValue += inc;
            return 1;
        }
        else {
            return 0;
        }
        
    }

    uint MSG_SpawnArrow(byte[] data)
    {
        GameObject arrowPrefab = Resources.Load<GameObject>("3D RightArrow");
        GameObject arrowInstance = GameObject.Instantiate(arrowPrefab, Vector3.zero, Quaternion.identity);

        // Set initial visibility of the arrow
        Renderer arrowRenderer = arrowInstance.GetComponent<Renderer>();
        if (arrowRenderer != null)
        {
            arrowRenderer.enabled = true; 
        }

   
       

        // Get the children of SelectedSetup
        Transform cArm = SelectedSetup.transform.GetChild(0); // C-Arm
        Transform laparoscopicTower4 = SelectedSetup.transform.GetChild(1); // LaparoscopicTower4
        Transform usTower = SelectedSetup.transform.GetChild(2); // USTower

        int index = BitConverter.ToInt32(data, 4);
        // Set the parent based on the index
        Transform parentTransform = null;
        switch (index)
        {
            case 0:
                parentTransform = cArm;
                
                break;
            case 1:
                parentTransform = laparoscopicTower4;
                
                break;
            case 2:
                parentTransform = usTower;
               
                break;
            default:
                Debug.LogError("Invalid index value!");
                return 0;
        }

        // Set the parent and ensure the transform is identity in the world
        arrowInstance.transform.SetParent(parentTransform, false);
        arrowInstance.transform.localPosition = Vector3.zero;
        arrowInstance.transform.localRotation = Quaternion.identity;
        arrowInstance.transform.localScale = Vector3.one;

        // Align local x-axis of the arrow prefab with the world y-axis
        arrowInstance.transform.rotation = Quaternion.FromToRotation(Vector3.right, Vector3.up); //not used i think




        return AddGameObject(arrowInstance);
    }

    // OK
    uint AddGameObject(GameObject go)
    {
        int key = go.GetInstanceID();
        m_remote_objects.Add(key, go);
        m_last_key = key;

        return (uint)key;
    }

    // OK
    int GetKey(byte[] data)
    {
        return m_mode ? m_last_key : BitConverter.ToInt32(data, 0);
    }

    // OK
    void UnpackTransform(byte[] data, int offset, out Vector3 position, out Quaternion rotation, out Vector3 locscale)
    {
        float[] f = new float[10];
        for (int i = 0; i < f.Length; ++i) { f[i] = BitConverter.ToSingle(data, offset + (i * 4)); }

        position = new Vector3(f[0], f[1], f[2]);
        rotation = new Quaternion(f[3], f[4], f[5], f[6]);
        locscale = new Vector3(f[7], f[8], f[9]);
    }

    // OK
    uint MSG_Remove(byte[] data)
    {
        if (data.Length < 4) { return 0; }

        GameObject go;
        int key = GetKey(data);
        if (!m_remote_objects.TryGetValue(key, out go)) { return 0; }

        m_remote_objects.Remove(key);
        Destroy(go);

        return 1;
    }

    // OK
    uint MSG_RemoveAll(byte[] data)
    {
        foreach (var go in m_remote_objects.Values) { Destroy(go); }
        m_remote_objects.Clear();
        return 1;
    }

    // OK
    uint MSG_BeginDisplayList(byte[] data)
    {
        m_loop = true;
        return 1;
    }

    // OK
    uint MSG_EndDisplayList(byte[] data)
    {
        m_loop = false;
        return 1;
    }

    // OK
    uint MSG_SetTargetMode(byte[] data)
    {
        if (data.Length < 4) { return 0; }
        m_mode = BitConverter.ToUInt32(data, 0) != 0;
        return 1;
    }

    // OK
    uint MSG_Disconnect(byte[] data)
    {
        m_loop = false;
        m_mode = false;
        m_last_key = 0;

        return ~0U;
    }

    // OK
    uint MSG_CreatePrimitive(byte[] data)
    {
        if (data.Length < 4) { return 0; }

        PrimitiveType t;
        switch (BitConverter.ToUInt32(data, 0))
        {
            case 0: t = PrimitiveType.Sphere; break;
            case 1: t = PrimitiveType.Capsule; break;
            case 2: t = PrimitiveType.Cylinder; break;
            case 3: t = PrimitiveType.Cube; break;
            case 4: t = PrimitiveType.Plane; break;
            default: t = PrimitiveType.Quad; break;
        }

        GameObject go = GameObject.CreatePrimitive(t);

        go.GetComponent<Renderer>().material = m_material;
        go.SetActive(false);

        return AddGameObject(go);
    }

    // OK
    uint MSG_SetActive(byte[] data)
    {
        if (data.Length < 8) { return 0; }

        GameObject go;
        if (!m_remote_objects.TryGetValue(GetKey(data), out go)) { return 0; }

        go.SetActive(BitConverter.ToInt32(data, 4) != 0);

        return 1;
    }

    // OK
    uint MSG_SetWorldTransform(byte[] data)
    {
        if (data.Length < 44) { return 0; }

        GameObject go;
        if (!m_remote_objects.TryGetValue(GetKey(data), out go)) { return 0; }

        Vector3 position;
        Quaternion rotation;
        Vector3 locscale;

        UnpackTransform(data, 4, out position, out rotation, out locscale);

        go.transform.parent = null;

        go.transform.SetPositionAndRotation(position, rotation);
        go.transform.localScale = locscale;

        return 1;
    }

    // OK
    uint MSG_SetLocalTransform(byte[] data)
    {
        if (data.Length < 44) { return 0; }

        GameObject go;
        if (!m_remote_objects.TryGetValue(GetKey(data), out go)) { return 0; }

        Vector3 position;
        Quaternion rotation;
        Vector3 locscale;

        UnpackTransform(data, 4, out position, out rotation, out locscale);

        go.transform.parent = transform;

        go.transform.localPosition = position;
        go.transform.localRotation = rotation;
        go.transform.localScale = locscale;

        return 1;
    }

    // OK
    uint MSG_SetColor(byte[] data)
    {
        if (data.Length < 20) { return 0; }

        GameObject go;
        if (!m_remote_objects.TryGetValue(GetKey(data), out go)) { return 0; }

        go.GetComponent<Renderer>().material.color = new Color(BitConverter.ToSingle(data, 4), BitConverter.ToSingle(data, 8), BitConverter.ToSingle(data, 12), BitConverter.ToSingle(data, 16));

        return 1;
    }

    // OK
    uint MSG_SetTexture(byte[] data)
    {
        if (data.Length < 4) { return 0; }

        GameObject go;
        if (!m_remote_objects.TryGetValue(GetKey(data), out go)) { return 0; }

        Texture2D tex;
        if (data.Length > 4)
        {
            tex = new Texture2D(2, 2);
            byte[] image = new byte[data.Length - 4];
            Array.Copy(data, 4, image, 0, image.Length);
            tex.LoadImage(image);
        }
        else
        {
            tex = null;
        }

        go.GetComponent<Renderer>().material.mainTexture = tex;

        return 1;
    }

    // OK
    uint MSG_CreateText(byte[] data)
    {
        GameObject go = new GameObject();
        TextMeshPro tmp = go.AddComponent<TextMeshPro>();

        go.SetActive(false);

        tmp.enableWordWrapping = false;
        tmp.autoSizeTextContainer = true;
        tmp.alignment = TextAlignmentOptions.Center;
        tmp.verticalAlignment = VerticalAlignmentOptions.Middle;
        tmp.text = "";

        return AddGameObject(go);
    }

    // OK
    uint MSG_SetText(byte[] data)
    {
        if (data.Length < 24) { return 0; }

        GameObject go;
        if (!m_remote_objects.TryGetValue(GetKey(data), out go)) { return 0; }
        TextMeshPro tmp = go.GetComponent<TextMeshPro>();
        if (tmp == null) { return 0; }

        tmp.fontSize = BitConverter.ToSingle(data, 4);
        tmp.color = new Color(BitConverter.ToSingle(data, 8), BitConverter.ToSingle(data, 12), BitConverter.ToSingle(data, 16), BitConverter.ToSingle(data, 20));

        string str;
        if (data.Length > 24)
        {
            byte[] str_bytes = new byte[data.Length - 24];
            Array.Copy(data, 24, str_bytes, 0, str_bytes.Length);
            try { str = System.Text.Encoding.UTF8.GetString(str_bytes); } catch { return 0; }
        }
        else
        {
            str = "";
        }

        tmp.text = str;

        return 1;
    }

    // OK
    uint MSG_Say(byte[] data)
    {
        string str;
        try { str = System.Text.Encoding.UTF8.GetString(data); } catch { return 0; }
        //TODO pass actual string currently default
        m_tts.GetComponent<TextToSpeech>().Speak(str);
        return 1;
    }
}
