using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SelectSetUp : MonoBehaviour
{
    // Public GameObjects to be assigned in the Unity Editor
    public GameObject gameObject1;
    public GameObject gameObject2;
    public GameObject gameObject3;
    public GameObject items;
    public List<GameObject> spheres;
    public RemoteUnitySceneCustom remoteUnitySceneCustom;

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }


    public void ActivateGameObject1()
    {
        remoteUnitySceneCustom.SelectedSetup = gameObject1;
        gameObject1.SetActive(true);
        gameObject2.SetActive(false);
        gameObject3.SetActive(false);
    }


    public void ActivateGameObject2()
    {
        gameObject2.SetActive(true);
        remoteUnitySceneCustom.SelectedSetup = gameObject2;
        gameObject1.SetActive(false);
        gameObject3.SetActive(false);
    }

    public void ActivateGameObject3()
    {
        gameObject3.SetActive(true);
        remoteUnitySceneCustom.SelectedSetup = gameObject3;
        gameObject1.SetActive(false);
        gameObject2.SetActive(false);
    }

    public void DoneSelectingSetUP()
    {
        items.SetActive(false);
        foreach (GameObject item in spheres) {

            item.SetActive(false);
        }

    }

    


}
