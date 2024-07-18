using MixedReality.Toolkit.Examples.Demos;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MixedReality.Toolkit.Examples.Demos;
using UnityEditor;

public class HandMenuManager : MonoBehaviour
{


    private int currentHandMenu = 0;
    private bool scanComplete = false;
    private bool findBedComplete = false;
    private bool selectSetUpComplete = false;
    private bool aiComplete = false;
    public List<GameObject> handMenus = new List<GameObject>();
    public RemoteUnitySceneCustom remoteUnitySceneCustom;

    private bool align1 = false;
    private bool align2 = false;
    private bool align3 = false;

    // Start is called before the first frame update
    void Start()
    {
        ToggleHandMenus(currentHandMenu);
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    public void AIDone()
    {
        if (!aiComplete)
        {

            aiComplete = true;
            currentHandMenu++;
            Debug.Log("aiComplete Complete");
            ToggleHandMenus(currentHandMenu);
        }


    }
    public void Align1Complete()
    {
        if (!align1)
        {
            align1 = true;
            remoteUnitySceneCustom.SetPathGroupActive(1);
            currentHandMenu++;
            ToggleHandMenus(currentHandMenu);
        }
    }

    public void Align2Complete()
    {
        if (!align2)
        {
            align2 = true;
            remoteUnitySceneCustom.SetPathGroupActive(2);
            currentHandMenu++;
            ToggleHandMenus(currentHandMenu);
        }
    }

    public void Align3Complete()
    {
        if (!align3)
        {
            align3 = true;
            remoteUnitySceneCustom.SetPathGroupActive(3);
            currentHandMenu++;
            ToggleHandMenus(currentHandMenu);
        }
    }

    public void SelectSetUpComplete()
    {
        if (!selectSetUpComplete)
        {

            findBedComplete = true;
            currentHandMenu++;
            Debug.Log("setup Complete");
            ToggleHandMenus(currentHandMenu);
        }


    }

    public void FindBedDone()
    {
        if (!findBedComplete)
        {
         
            findBedComplete = true;
            currentHandMenu++;
            Debug.Log("find bed Complete");
            ToggleHandMenus(currentHandMenu);
        }


    }



    public void ScanDone() {
        if (!scanComplete) {
            // under threshold display warning TODO
            scanComplete = true;
            //else load next hand menu
            currentHandMenu++;
            Debug.Log("ScanComplete");
            ToggleHandMenus(currentHandMenu);
        }


    }

    private void ToggleHandMenus(int index) {
        foreach (var menu in handMenus)
        {
            menu.SetActive(false);
        }
        handMenus[index].SetActive(true);

    }
}
