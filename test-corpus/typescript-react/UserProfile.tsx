import { useState, useEffect, useCallback, useMemo } from 'react';

export type User = {
  id: string;
  name: string;
  email: string;
  joinedAt: Date;
};

type UserProfileProps = {
  userId: string;
  onUpdate?: (user: User) => void;
};

async function fetchUser(userId: string, signal: AbortSignal): Promise<User> {
  const response = await fetch(`/api/users/${userId}`, { signal });
  if (!response.ok) {
    throw new Error(`Failed to load user ${userId}: ${response.status}`);
  }
  const data = await response.json();
  return { ...data, joinedAt: new Date(data.joinedAt) };
}

export function UserProfile({ userId, onUpdate }: UserProfileProps) {
  const [user, setUser] = useState<User | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    fetchUser(userId, controller.signal)
      .then((u) => {
        setUser(u);
        setError(null);
      })
      .catch((e) => {
        if (e.name !== 'AbortError') setError(e);
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [userId]);

  const handleSave = useCallback(
    async (next: Partial<User>) => {
      if (!user) return;
      const merged = { ...user, ...next };
      setUser(merged);
      onUpdate?.(merged);
    },
    [user, onUpdate],
  );

  const memberSince = useMemo(
    () => (user ? user.joinedAt.toLocaleDateString() : ''),
    [user],
  );

  if (loading) return <div role="status">Loading…</div>;
  if (error) return <div role="alert">Error: {error.message}</div>;
  if (!user) return null;

  return (
    <article>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
      <p>Member since {memberSince}</p>
      <button onClick={() => handleSave({ name: 'updated' })}>Save</button>
    </article>
  );
}
