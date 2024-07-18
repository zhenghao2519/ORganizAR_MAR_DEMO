using UnityEngine;

public class TableRendererController : MonoBehaviour
{

    public GameObject SetUp1;
    public void EnableAllMeshRenderers()
    {
        SetUp1.SetActive(true);
        this.GetComponent<MeshRenderer>().enabled = true;
        /*GameObject table = this.gameObject;
        if (table != null)
        {
            ToggleMeshRenderer(table, true);
            foreach (Transform child in table.transform)
            {
                ToggleMeshRenderer(child.gameObject, true);
            }
        }
        else
        {
            Debug.LogError("Table GameObject is not assigned.");
        }*/
    }

    public void DisableTable() {
        SetUp1.SetActive(false);
        this.GetComponent<MeshRenderer>().enabled = false;
    }

    private void ToggleMeshRenderer(GameObject obj, bool state)
    {
        /*MeshRenderer renderer = obj.GetComponent<MeshRenderer>();
        if (renderer != null)
        {
            renderer.enabled = state;
        }

        foreach (Transform child in obj.transform)
        {
            ToggleMeshRenderer(child.gameObject, state);
        }*/
    }
}
