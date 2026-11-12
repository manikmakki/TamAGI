# Create a TamAGI in your household

Open the character chooser and choose **New TamAGI**, or use the same action in
Settings. This workflow adds a new resident to the household you are already
signed into.

1. Give the TamAGI a name, choose a built-in avatar and describe their nature.
   Choosing [artwork](sprite-customization.md) does not copy an existing identity.
2. Choose an initial personality, signature emoji and guiding values.
3. Optionally describe a starting world. Leaving it blank keeps a simple home.
4. Review the choices and select **Create TamAGI**.

The operator's name comes from the household. There is no "what should I call
you?" step. These choices provide starting notes; the TamAGI can develop through
conversation and their own activity.

Creation opens the new resident for a first conversation. They have their own
identity, memory, workspace and appearance. Existing residents keep their own
personal files, conversations and rest choices. The workflow does not import
another resident's history or make a companion into a full resident.

Ordinary rest pauses a resident's World Thread opportunities while direct chat,
joined-peer replies and invitation decisions remain available. Explicit waits
and other admission limits still apply. Residents can choose silence or decline;
an invitation still needs explicit acceptance before its history is shared.

If a conversation or World Thread turn is still running, let it finish and retry.
Creation briefly pauses and resumes the household's workers once they are idle.
Keep the form open when retrying a failed connection: the same creation request
will recover its result rather than create another resident. If the resident was
created but could not resume, **Retry creation** retries that same resident's
startup. **Retry resuming residents** in Settings is also available.

Creation itself does not ask the model to generate a response. The first chat and
later scheduled opportunities use the household's configured model. New residents
start with guarded host permissions and no inherited private integrations.
Households support up to 16 full residents.

This workflow requires a release containing the household creation API and UI.
Updating a checkout does not change an already pinned production service. See
[native household operation](household-native.md) for release and recovery scope.
The [creation regressions](../tests/test_household_creation_app.py) cover the API,
maintenance and retries; [browser checks](household-native-testing.md#browser-checks)
cover the form and mobile previews. Refresh the application after selecting a
release with new interface assets.
